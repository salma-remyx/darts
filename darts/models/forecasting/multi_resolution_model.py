"""
Multi-Resolution Residual-Correction Model
------------------------------------------

Combine a low-frequency base forecaster with a high-frequency residual forecaster.
"""

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.models.forecasting.forecasting_model import (
    GlobalForecastingModel,
    LocalForecastingModel,
)
from darts.utils.timeseries_generation import _build_forecast_series
from darts.utils.ts_utils import get_single_series
from darts.utils.utils import n_steps_between

logger = get_logger(__name__)


class MultiResolutionModel(GlobalForecastingModel):
    def __init__(
        self,
        base_model,
        residual_model,
        base_freq: str | None = None,
        base_agg: str = "mean",
        residual_fit_length: int | None = None,
    ):
        """Multi-Resolution Residual-Correction forecasting model.

        Combines two models operating at different temporal resolutions [1]_:

        - a ``base_model`` forecasting the *coarse* (low-frequency) signal, trained on the target
          series aggregated to ``base_freq``
        - a ``residual_model`` correcting the base forecast at the *native* (high) frequency, trained
          on the base model's recent errors: the high-frequency signal the aggregation discards
          (intra-period shapes, short spikes, ...)

        Producing a forecast for `n` native steps after the end of `series` goes as follows:

        1. Aggregate the target to ``base_freq`` and forecast the coarse steps covering the horizon.
        2. Broadcast the coarse forecast back onto the native time index (each native step inherits
           the value of the coarse block it falls in).
        3. Fit the ``residual_model`` on the residuals (actual - base backcast) of the most recent
           ``residual_fit_length`` native steps, and add its `n`-step forecast on top.

        The base model is fitted eagerly in ``fit()``; the residual model is fitted lazily, on the
        series passed to `predict()` (or the training series if none). This mirrors the zero-shot
        usage in [1]_, where a pre-trained foundation model provides the coarse forecast and only the
        light residual correction is fitted locally.

        Parameters
        ----------
        base_model
            An untrained forecasting model for the coarse signal.
        residual_model
            An untrained forecasting model for the native-frequency residuals.
        base_freq
            The aggregation frequency for the base model, as a pandas offset alias (e.g. ``"4h"``
            for four-hourly blocks of an hourly series). Must yield a coarser index than the
            target's own frequency. Default: ``None`` (no aggregation, both models run at the
            native frequency).
        base_agg
            Aggregation method applied when down-sampling the target to ``base_freq``. Passed to
            :meth:`TimeSeries.resample() <darts.timeseries.TimeSeries.resample>`. Default: ``"mean"``.
        residual_fit_length
            Optionally, the number of most recent native steps used to fit the residual model.
            If ``None``, the whole series is used. Default: ``None``.

        References
        ----------
        .. [1] "Systematic Evaluation of TabPFN-TS for Zero-Shot Probabilistic Heat Load Forecasting
               in District Heating Networks", arXiv:2608.20024, 2026.

        Examples
        --------
        >>> from darts.datasets import AirPassengersDataset
        >>> from darts.models import LinearRegressionModel, MultiResolutionModel, NaiveSeasonal
        >>> series = AirPassengersDataset().load()
        >>> model = MultiResolutionModel(
        ...     base_model=NaiveSeasonal(K=4),
        ...     residual_model=LinearRegressionModel(lags=4),
        ...     base_freq="4MS",
        ... )
        >>> model.fit(series)
        >>> pred = model.predict(6)
        """
        for name, model in (
            ("base_model", base_model),
            ("residual_model", residual_model),
        ):
            if isinstance(model, MultiResolutionModel):
                raise_log(
                    ValueError(f"`{name}` cannot be a `MultiResolutionModel` itself."),
                )
            if not isinstance(model, GlobalForecastingModel | LocalForecastingModel):
                raise_log(
                    ValueError(f"`{name}` must be a darts forecasting model."),
                )
            if model._fit_called:
                raise_log(
                    ValueError(f"`{name}` must be an untrained model."),
                )
        if base_freq is not None:
            # raises for invalid aliases
            pd.tseries.frequencies.to_offset(base_freq)
        if residual_fit_length is not None and residual_fit_length < 1:
            raise_log(
                ValueError("`residual_fit_length` must be `>=1` or `None`."),
            )

        super().__init__(add_encoders=None)

        self.base_model = base_model
        self.residual_model = residual_model
        self.base_freq = base_freq
        self.base_agg = base_agg
        self.residual_fit_length = residual_fit_length
        self._stride: int = 1
        self._uses_covariates = False

    @property
    def _has_local_models(self) -> bool:
        return isinstance(self.base_model, LocalForecastingModel) or isinstance(
            self.residual_model, LocalForecastingModel
        )

    def fit(
        self,
        series,
        past_covariates=None,
        future_covariates=None,
        sample_weight=None,
        verbose=None,
    ):
        """Fit the base model on the aggregated target.

        The residual model is not fitted here; it is fitted lazily on the residuals of the series
        available at prediction time (see :func:`predict()`).

        Parameters
        ----------
        series
            A single target time series with a datetime index. A sequence of series is only
            supported if both sub-models are `GlobalForecastingModels`.
        past_covariates
            Optionally, a past-observed covariates series. Passed to the sub-models supporting them.
        future_covariates
            Optionally, a future-known covariates series. Passed to the sub-models supporting them.
        sample_weight
            Optionally, sample weights. Passed to the sub-models supporting them.
        verbose
            Optionally, set the fit verbosity. Not effective for all models.

        Returns
        -------
        self
            Fitted model.
        """
        series = self._check_input_series(series)
        past_covariates = self._check_input_covariates(past_covariates)
        future_covariates = self._check_input_covariates(future_covariates)

        self._setup_resolution(series)

        super().fit(
            series=series,
            past_covariates=past_covariates,
            future_covariates=future_covariates,
        )

        self.base_model._fit_wrapper(
            series=self._aggregate(series),
            past_covariates=self._covs_for(past_covariates, self.base_model, "past"),
            future_covariates=self._covs_for(
                future_covariates, self.base_model, "future"
            ),
            sample_weight=(
                sample_weight if self.base_model.supports_sample_weight else None
            ),
            verbose=verbose,
        )
        self._uses_covariates = (
            self.base_model.uses_past_covariates
            or self.base_model.uses_future_covariates
        )
        return self

    def predict(
        self,
        n: int,
        series=None,
        past_covariates=None,
        future_covariates=None,
        num_samples: int = 1,
        verbose=None,
        predict_likelihood_parameters: bool = False,
        show_warnings: bool = True,
        random_state: int | None = None,
    ):
        """Forecasts values for `n` time steps after the end of the (input) series.

        The base model provides the coarse forecast, which is broadcast to the native frequency and
        corrected by the residual model's forecast. The residual model is (re-)fitted on the
        residuals of the input `series` before predicting.

        Parameters
        ----------
        n
            Forecast horizon - the number of native-frequency time steps after the end of the series
            for which to produce predictions.
        series
            Optionally, the series whose future values will be predicted. If not provided, the
            training series is used.
        past_covariates
            Optionally, a past-observed covariates series. Passed to the sub-models supporting them.
        future_covariates
            Optionally, a future-known covariates series. Passed to the sub-models supporting them.
        num_samples
            Number of times a prediction is sampled from a probabilistic model. Must be `1` for
            deterministic models.
        verbose
            Optionally, set the prediction verbosity. Not effective for all models.
        predict_likelihood_parameters
            Not supported by this model. Must be ``False``.
        show_warnings
            Whether to show warnings related auto-regression and past covariates usage.
        random_state
            Controls the randomness for probabilistic predictions.

        Returns
        -------
        TimeSeries
            The `n` next points after the end of the (input) series.
        """
        super().predict(n, num_samples, verbose=verbose, random_state=random_state)
        if predict_likelihood_parameters:
            raise_log(
                ValueError(
                    "`predict_likelihood_parameters=True` is not supported by "
                    "`MultiResolutionModel`."
                ),
            )

        series = self.training_series if series is None else series
        series = self._check_input_series(series)

        residuals = self._get_residuals(series)
        self.residual_model._fit_wrapper(
            series=residuals,
            verbose=verbose,
        )

        low_pred = self.base_model._predict_wrapper(
            n=self._low_horizon(n, series),
            series=self._aggregate(series),
            past_covariates=self._covs_for(past_covariates, self.base_model, "past"),
            future_covariates=self._covs_for(
                future_covariates, self.base_model, "future"
            ),
            verbose=verbose,
        )
        base_pred = self._upsample_forecast(low_pred, series, n)

        residual_pred = self.residual_model._predict_wrapper(
            n=n,
            series=residuals,
            past_covariates=self._covs_for(
                past_covariates, self.residual_model, "past"
            ),
            future_covariates=self._covs_for(
                future_covariates, self.residual_model, "future"
            ),
            num_samples=(
                num_samples
                if self.residual_model.supports_probabilistic_prediction
                else 1
            ),
            random_state=random_state,
            verbose=verbose,
        )
        return base_pred + residual_pred

    @staticmethod
    def _covs_for(covariates, model, covs_type: str):
        """Pass covariates to a sub-model only if it supports them."""
        supported = getattr(model, f"supports_{covs_type}_covariates")
        return covariates if supported else None

    # --- resolution handling ---

    def _setup_resolution(self, series: TimeSeries) -> None:
        """Validate the target index and derive the native stride between coarse blocks."""
        if series.has_range_index:
            raise_log(
                ValueError(
                    "`MultiResolutionModel` requires a `TimeSeries` with a datetime index."
                ),
            )
        if self.base_freq is None:
            self._stride = 1
            return
        offset = pd.tseries.frequencies.to_offset(self.base_freq)
        steps = n_steps_between(
            end=series.start_time() + offset,
            start=series.start_time(),
            freq=series.freq,
        )
        if steps < 1:
            raise_log(
                ValueError(
                    f"`base_freq='{self.base_freq}'` ({offset}) must be a whole multiple of "
                    f"the target series frequency ({series.freq})."
                ),
            )
        self._stride = steps

    def _aggregate(self, series: TimeSeries) -> TimeSeries:
        """Down-sample `series` to the base model's coarse frequency."""
        if self._stride == 1:
            return series
        return series.resample(self.base_freq, method=self.base_agg)

    def _low_horizon(self, n: int, series: TimeSeries) -> int:
        """Number of coarse steps covering the `n` native steps after the end of `series`."""
        if self._stride == 1:
            return n
        # blocks are anchored on the target's own grid: the block containing the last training
        # timestamp covers the `stride` native steps ending there
        return -(-n // self._stride)

    def _upsample_forecast(
        self, low_pred: TimeSeries, series: TimeSeries, n: int
    ) -> TimeSeries:
        """Broadcast a coarse forecast onto the `n` native steps after the end of `series`."""
        native_index = pd.date_range(
            start=series.end_time() + series.freq,
            periods=n,
            freq=series.freq,
            name=series.time_index.name,
        )
        if self._stride == 1:
            return _build_forecast_series(
                low_pred.values(copy=False)[:n],
                series,
                time_index=native_index,
            )
        # each native step inherits the value of the coarse block it falls in; the coarse blocks are
        # anchored on the target's own grid, so the last training timestamp closes block
        # `(len(series) - 1) // stride` and the first forecast block is the next one
        remainder = len(series) % self._stride
        # native steps remaining in the (possibly partial) block that the last training timestamp
        # belongs to; the forecast starts right after them
        steps_to_next_block = (self._stride - remainder) % self._stride
        first_block = (len(series) + steps_to_next_block) // self._stride
        native_pos = np.arange(n) + steps_to_next_block
        block_idx = first_block + native_pos // self._stride
        values = np.asarray(low_pred.values(copy=False))
        upsampled = values[np.minimum(block_idx, len(values) - 1)]
        return _build_forecast_series(upsampled, series, time_index=native_index)

    def _get_residuals(self, series: TimeSeries) -> TimeSeries:
        """Base-model residuals (actual - backcast) at the native frequency."""
        low_series = self._aggregate(series)
        low_backcast = self.base_model._predict_wrapper(
            n=len(low_series), series=low_series
        )
        # anchor the backcast on the native grid, aligned with the blocks it was aggregated from
        backcast_index = pd.date_range(
            start=series.start_time(),
            periods=len(low_backcast),
            freq=low_series.freq,
            name=series.time_index.name,
        )
        backcast = _build_forecast_series(
            low_backcast.values(copy=False),
            series,
            time_index=backcast_index,
        )
        # expand each coarse value to the `stride` native steps of its block
        expanded = np.repeat(backcast.values(copy=False), self._stride, axis=0)
        actual_values = series.values(copy=False)
        n_overlap = min(len(expanded), len(actual_values))
        residuals = actual_values[:n_overlap] - expanded[:n_overlap]
        residual_index = series.time_index[:n_overlap]
        residual_series = TimeSeries.from_times_and_values(
            residual_index,
            residuals,
            columns=series.components,
            static_covariates=series.static_covariates,
        )
        if self.residual_fit_length is not None:
            residual_series = residual_series[-self.residual_fit_length :]
        return residual_series

    def _check_input_series(self, series) -> TimeSeries:
        if self._has_local_models and not isinstance(series, TimeSeries):
            raise_log(
                ValueError(
                    "The sub-models contain a `LocalForecastingModel`, which does not support "
                    "multiple series."
                ),
            )
        if isinstance(series, TimeSeries):
            return series
        return get_single_series(series)

    def _check_input_covariates(self, covariates):
        if covariates is not None and self._has_local_models:
            raise_log(
                ValueError(
                    "Covariates are not supported with `LocalForecastingModel` sub-models."
                ),
            )
        return covariates

    # --- model capabilities, derived from the sub-models ---

    @property
    def _model_encoder_settings(self):
        raise NotImplementedError(
            "Encoders are not supported by `MultiResolutionModel`. Instead add encoders to the "
            "underlying `base_model` or `residual_model`."
        )

    @property
    def extreme_lags(
        self,
    ) -> tuple[
        int | None, int | None, int | None, int | None, int | None, int | None, int
    ]:
        return self.base_model.extreme_lags

    @property
    def _target_window_lengths(self) -> tuple[int, int]:
        return self.residual_model._target_window_lengths

    @property
    def min_train_samples(self) -> int:
        return self.residual_model.min_train_samples

    @property
    def output_chunk_length(self) -> int | None:
        # the native-horizon output is limited by the residual model
        return self.residual_model.output_chunk_length

    @property
    def output_chunk_shift(self) -> int:
        return self.residual_model.output_chunk_shift

    @property
    def supports_multivariate(self) -> bool:
        return (
            self.base_model.supports_multivariate
            and self.residual_model.supports_multivariate
        )

    @property
    def supports_past_covariates(self) -> bool:
        return (
            self.base_model.supports_past_covariates
            or self.residual_model.supports_past_covariates
        )

    @property
    def supports_future_covariates(self) -> bool:
        return (
            self.base_model.supports_future_covariates
            or self.residual_model.supports_future_covariates
        )

    @property
    def supports_static_covariates(self) -> bool:
        return (
            self.base_model.supports_static_covariates
            and self.residual_model.supports_static_covariates
        )

    @property
    def supports_sample_weight(self) -> bool:
        return (
            self.base_model.supports_sample_weight
            and self.residual_model.supports_sample_weight
        )

    @property
    def supports_probabilistic_prediction(self) -> bool:
        return self.residual_model.supports_probabilistic_prediction

    @property
    def uses_past_covariates(self) -> bool:
        return self.residual_model.uses_past_covariates

    @property
    def uses_future_covariates(self) -> bool:
        return self.residual_model.uses_future_covariates

    @property
    def uses_static_covariates(self) -> bool:
        return self.residual_model.uses_static_covariates

    @property
    def considers_static_covariates(self) -> bool:
        return self.residual_model.considers_static_covariates
