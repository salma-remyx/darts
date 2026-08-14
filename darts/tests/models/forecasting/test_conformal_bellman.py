import numpy as np
import pytest

from darts import TimeSeries
from darts.metrics import mic
from darts.models import ConformalNaiveModel, LinearRegressionModel
from darts.models.forecasting.conformal_bellman import (
    ConformalBellmanModel,
    bellman_interval_widths,
)
from darts.utils import timeseries_generation as tg
from darts.utils.likelihood_models.base import (
    LikelihoodType,
    likelihood_component_names,
    quantile_names,
)

q = [0.1, 0.25, 0.5, 0.75, 0.9]


def _trained_model_and_series(n=300, seed=42):
    """LinearRegressionModel on heteroscedastic noise: interval width cost varies
    strongly across the horizon, which is where joint budget allocation pays off."""
    rng = np.random.default_rng(seed)
    base = tg.sine_timeseries(length=n, value_y_offset=10.0)
    noise = rng.normal(0.0, 1.0, n) * np.linspace(0.5, 3.0, n)
    series = base + TimeSeries.from_values(
        noise.reshape(-1, 1), columns=base.columns.tolist()
    )
    model = LinearRegressionModel(lags=6, output_chunk_length=5).fit(series)
    return model, series


class TestConformalBellmanModel:
    def test_construction(self):
        model, series = _trained_model_and_series()
        cm = ConformalBellmanModel(model=model, quantiles=q)
        assert cm.likelihood.type is LikelihoodType.Quantile
        # target miscoverage comes from the widest interval (0.9 - 0.1 = 0.8 coverage)
        assert cm.alpha0 == pytest.approx(0.2)

        with pytest.raises(ValueError) as exc:
            ConformalBellmanModel(model=model, quantiles=q, symmetric=False)
        assert str(exc.value) == (
            "`ConformalBellmanModel` only supports `symmetric=True` "
            "(symmetric non-conformity scores)."
        )

        with pytest.raises(ValueError):
            ConformalBellmanModel(model=model, quantiles=q, lam=-1.0)

    def test_predict_quantile_parameters(self):
        model, series = _trained_model_and_series()
        cm = ConformalBellmanModel(model=model, quantiles=q, cal_length=40)

        preds = cm.predict(
            n=5, series=series, num_samples=1, predict_likelihood_parameters=True
        )
        assert preds.columns.tolist() == likelihood_component_names(
            series.columns, quantile_names(q)
        )
        assert len(preds) == 5
        assert preds.start_time() == series.end_time() + series.freq

        # intervals are ordered (lower < median < upper) and nested per interval
        vals = preds.values().reshape(len(preds), -1, len(q))
        assert np.all(np.diff(vals, axis=2) >= -1e-9)

    def test_budget_allocation_beats_fixed_quantiles(self):
        """The paper's core result: at matched long-run coverage, jointly
        allocating the miscoverage budget gives shorter total interval length
        than calibrating each horizon step independently."""
        model, series = _trained_model_and_series()
        kwargs = dict(cal_length=50)
        cm_bellman = ConformalBellmanModel(model=model, quantiles=q, **kwargs)
        cm_naive = ConformalNaiveModel(model=model, quantiles=q, **kwargs)

        preds_bellman = cm_bellman.predict(
            n=10, series=series, num_samples=1, predict_likelihood_parameters=True
        )
        preds_naive = cm_naive.predict(
            n=10, series=series, num_samples=1, predict_likelihood_parameters=True
        )

        width = lambda p: float((p.values()[:, -1] - p.values()[:, 0]).sum())
        total_bellman, total_naive = width(preds_bellman), width(preds_naive)

        # backtest the innermost-interval coverage of both models on held-out past
        hfc_kwargs = dict(
            series=series,
            forecast_horizon=10,
            start=210,
            stride=5,
            retrain=False,
            last_points_only=True,
            num_samples=1,
            predict_likelihood_parameters=True,
        )
        cov_bellman = mic(
            series,
            cm_bellman.historical_forecasts(**hfc_kwargs),
            q_interval=[(0.1, 0.9)],
        )
        cov_naive = mic(
            series,
            cm_naive.historical_forecasts(**hfc_kwargs),
            q_interval=[(0.1, 0.9)],
        )

        # comparable coverage (within 10pp), strictly shorter total width
        assert abs(cov_bellman - cov_naive) <= 0.1
        assert total_bellman < total_naive

    def test_explicit_lam_favors_coverage(self):
        """Increasing the terminal cost weight must widen intervals (the knob the
        paper exposes for trading width against coverage)."""
        model, series = _trained_model_and_series()
        widths = []
        for lam in [0.1, 10.0]:
            cm = ConformalBellmanModel(model=model, quantiles=q, cal_length=50, lam=lam)
            preds = cm.predict(
                n=5, series=series, num_samples=1, predict_likelihood_parameters=True
            )
            widths.append(float((preds.values()[:, -1] - preds.values()[:, 0]).sum()))
        assert widths[1] >= widths[0]

    def test_bellman_interval_widths_inputs(self):
        scores = np.abs(np.random.default_rng(0).normal(0, 1, (4, 20)))
        with pytest.raises(ValueError):
            bellman_interval_widths(scores, alpha0=0.0)
        with pytest.raises(ValueError):
            bellman_interval_widths(scores[:, :0], alpha0=0.2)
        # higher coverage target gives (weakly) wider intervals
        w_lo = bellman_interval_widths(scores, alpha0=0.5)
        w_hi = bellman_interval_widths(scores, alpha0=0.1)
        assert np.all(w_hi >= w_lo - 1e-9)
