import numpy as np
import pytest

from darts.models import (
    ConformalBellmanModel,
    ConformalNaiveModel,
    LinearRegressionModel,
)
from darts.models.forecasting.bellman_conformal import bellman_interval_radii
from darts.utils import timeseries_generation as tg

IN_LEN = 3
OUT_LEN = 3
regr_kwargs = {"lags": IN_LEN, "output_chunk_length": OUT_LEN}
q = [0.1, 0.5, 0.9]


class TestConformalBellmanModel:
    def test_model_construction(self):
        series = tg.sine_timeseries(length=30)
        with pytest.raises(ValueError) as exc:
            ConformalBellmanModel(model=LinearRegressionModel(**regr_kwargs), quantiles=q)
        assert str(exc.value) == "`model` must be a pre-trained `GlobalForecastingModel`."

        model = ConformalBellmanModel(
            model=LinearRegressionModel(**regr_kwargs).fit(series), quantiles=q
        )
        assert model.symmetric
        for param, value in (
            ("step_size", 0.05),
            ("max_weight", 50.0),
            ("weight_window", 100),
        ):
            assert getattr(model, param) == value

        with pytest.raises(ValueError):
            ConformalBellmanModel(
                model=LinearRegressionModel(**regr_kwargs).fit(series),
                quantiles=q,
                step_size=0.0,
            )
        with pytest.raises(ValueError):
            ConformalBellmanModel(
                model=LinearRegressionModel(**regr_kwargs).fit(series),
                quantiles=q,
                weight_window=0,
            )

    def test_predict_quantile_output(self):
        # exercises the `_calibrate_interval` hook wired into the conformal calibration loop
        series = (
            tg.sine_timeseries(length=120)
            + tg.linear_timeseries(length=120) * 0.1
            + tg.gaussian_timeseries(length=120, std=0.5)
        )
        model = LinearRegressionModel(**regr_kwargs, random_state=42).fit(series)
        pred = ConformalBellmanModel(model=model, quantiles=q).predict(
            n=OUT_LEN, num_samples=1, predict_likelihood_parameters=True
        )
        assert pred.components.tolist() == [
            "sine_q0.100",
            "sine_q0.500",
            "sine_q0.900",
        ]
        values = pred.values()
        # intervals must contain the median and be ordered
        assert np.all(values[:, 0] <= values[:, 1])
        assert np.all(values[:, 1] <= values[:, 2])
        # at least one interval must be non-degenerate (scores are not all identical)
        assert np.any(values[:, 2] - values[:, 0] > 0.0)

        # samples can be drawn from the calibrated intervals
        samples = ConformalBellmanModel(model=model, quantiles=q).predict(
            n=OUT_LEN, num_samples=50, random_state=42
        )
        assert samples.n_samples == 50

    def test_historical_forecasts(self):
        series = (
            tg.sine_timeseries(length=120) + tg.gaussian_timeseries(length=120, std=0.5)
        )
        model = LinearRegressionModel(**regr_kwargs, random_state=42).fit(series)
        hfc = ConformalBellmanModel(model=model, quantiles=q).historical_forecasts(
            series,
            num_samples=1,
            predict_likelihood_parameters=True,
            forecast_horizon=OUT_LEN,
            stride=1,
            last_points_only=True,
            retrain=False,
        )
        assert len(hfc[0]) > 0
        assert hfc[0].n_components == len(q)

    def test_bellman_vs_naive(self):
        # both models wrap the same underlying model; the bellman model reallocates interval
        # length across the horizon but must not be systematically wider or unordered
        series = (
            tg.sine_timeseries(length=120) + tg.gaussian_timeseries(length=120, std=0.5)
        )
        model = LinearRegressionModel(**regr_kwargs, random_state=42).fit(series)
        preds = [
            cls(model=model, quantiles=q).predict(
                n=OUT_LEN, num_samples=1, predict_likelihood_parameters=True
            )
            for cls in (ConformalNaiveModel, ConformalBellmanModel)
        ]
        widths = [p.values()[:, 2] - p.values()[:, 0] for p in preds]
        # total width within an order of magnitude of the static conformal model
        assert 0.5 < widths[1].sum() / widths[0].sum() < 2.0

    def test_radii_calibration_behavior(self):
        scores = np.abs(np.random.default_rng(0).normal(size=(3, 200)))

        # i.i.d. scores: the planned radii stay close to the static split-conformal quantile
        radii = bellman_interval_radii(scores, miscoverage=0.1)
        static = np.quantile(scores, 0.9, axis=1, method="higher")
        assert np.allclose(radii, static, rtol=0.25)

        # recent under-coverage widens the intervals
        shifted = scores.copy()
        shifted[:, -15:] *= 3.0
        assert np.all(
            bellman_interval_radii(shifted, miscoverage=0.1)
            >= bellman_interval_radii(scores, miscoverage=0.1) - 1e-12
        )

        # wider target coverage gives narrower intervals
        narrow = bellman_interval_radii(scores, miscoverage=0.5)
        assert np.all(narrow <= radii)

    def test_radii_input_checks(self):
        with pytest.raises(ValueError):
            bellman_interval_radii(np.ones((2, 5)), miscoverage=0.0)
        with pytest.raises(ValueError):
            bellman_interval_radii(np.ones((2, 5)), miscoverage=1.0)
