import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.metrics import mae
from darts.models import (
    LinearRegressionModel,
    MultiResolutionModel,
    NaiveMean,
    NaiveSeasonal,
    SKLearnModel,
)
from darts.models.forecasting.multi_resolution_model import (
    MultiResolutionModel as MultiResolutionModelImpl,
)
from darts.tests.models.forecasting.test_sklearn_models import train_test_split


def _heat_load_like(n_days: int = 60, seed: int = 7) -> TimeSeries:
    """Hourly series with a daily cycle on top of a slower weekly cycle (heat-load-like)."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n_days * 24, freq="h", name="time")
    hour = np.arange(len(idx))
    values = (
        5.0
        + 1.0 * np.sin(2 * np.pi * hour / (24 * 7))
        + 2.0 * np.sin(2 * np.pi * hour / 24 - np.pi / 2)
        + rng.normal(0, 0.2, len(idx))
    )
    return TimeSeries.from_times_and_values(idx, values, columns=["load"])


class TestMultiResolutionModel:
    series = _heat_load_like()

    def test_fit_predict(self):
        """The composed forecast is aligned with the target grid and tracks the held-out signal."""
        model = MultiResolutionModel(
            base_model=NaiveSeasonal(K=7),
            residual_model=LinearRegressionModel(lags=24),
            base_freq="24h",
        )
        assert not model.residual_model._fit_called
        train, test = train_test_split(self.series, pd.Timestamp("2020-02-28"))
        model.fit(train)

        # the base model is fitted eagerly, the residual model lazily at predict time
        assert model.base_model._fit_called
        assert not model.residual_model._fit_called

        pred = model.predict(len(test))
        assert model.residual_model._fit_called
        assert pred.start_time() == train.end_time() + train.freq
        assert pred.end_time() == test.end_time()
        assert pred.time_index.equals(test.time_index)
        assert pred.freq == train.freq
        # the daily cycle is by far the dominant term of the signal
        assert mae(test, pred) < 1.0

    def test_residual_correction_improves_base(self):
        """Adding the residual correction must not be worse than the coarse base forecast alone."""
        model = MultiResolutionModel(
            base_model=NaiveMean(),
            residual_model=LinearRegressionModel(lags=24),
            base_freq="24h",
        )
        train, test = train_test_split(self.series, pd.Timestamp("2020-02-28"))
        model.fit(train)
        pred = model.predict(24)

        base_only = NaiveMean().fit(train).predict(24)
        assert mae(test, pred) < mae(test, base_only)

    def test_residuals_are_aligned(self):
        """For a constant target, the base backcast reproduces the target -> residuals are zero."""
        idx = pd.date_range("2020-01-01", periods=24 * 5 + 7, freq="h")
        const = TimeSeries.from_times_and_values(idx, np.full(len(idx), 3.0))
        model = MultiResolutionModel(
            base_model=NaiveMean(),
            residual_model=LinearRegressionModel(lags=2),
            base_freq="24h",
        )
        model.fit(const)
        residuals = model._get_residuals(const)
        assert len(residuals) == len(const)
        assert np.allclose(residuals.values(copy=False), 0.0)

    def test_predict_with_new_series(self):
        """Global sub-models allow forecasting an unseen series."""
        model = MultiResolutionModel(
            base_model=SKLearnModel(lags=24),
            residual_model=SKLearnModel(lags=6),
            base_freq="24h",
        )
        other = _heat_load_like(seed=13)
        model.fit([self.series, other])
        unseen = _heat_load_like(seed=21)
        pred = model.predict(12, series=unseen)
        assert pred.start_time() == unseen.end_time() + unseen.freq
        assert len(pred) == 12

    def test_input_checks(self):
        with pytest.raises(ValueError, match="untrained"):
            MultiResolutionModel(
                base_model=NaiveMean().fit(self.series),
                residual_model=LinearRegressionModel(lags=2),
            )
        with pytest.raises(ValueError, match="cannot be a `MultiResolutionModel`"):
            MultiResolutionModel(
                base_model=MultiResolutionModel(
                    base_model=NaiveMean(),
                    residual_model=LinearRegressionModel(lags=2),
                ),
                residual_model=LinearRegressionModel(lags=2),
            )
        with pytest.raises(ValueError, match="whole multiple"):
            model = MultiResolutionModel(
                base_model=NaiveMean(),
                residual_model=LinearRegressionModel(lags=2),
                base_freq="30min",
            )
            model.fit(self.series)
        with pytest.raises(ValueError, match="datetime index"):
            model = MultiResolutionModel(
                base_model=SKLearnModel(lags=2),
                residual_model=SKLearnModel(lags=2),
                base_freq="2h",
            )
            model.fit(TimeSeries.from_values(np.arange(20.0)))

    def test_public_api_matches_impl(self):
        """The model is exported from `darts.models` under its capability name."""
        assert MultiResolutionModel is MultiResolutionModelImpl
