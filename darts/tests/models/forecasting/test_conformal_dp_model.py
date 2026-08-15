import numpy as np
import pytest

from darts.metrics import ae
from darts.models import ConformalDPModel, ConformalNaiveModel, LinearRegressionModel
from darts.tests.models.forecasting.test_conformal_model import (
    OUT_LEN,
    train_model,
)
from darts.utils import timeseries_generation as tg

q = [0.1, 0.5, 0.9]
pred_lklp = {"num_samples": 1, "predict_likelihood_parameters": True}


class TestConformalDPModel:
    def test_model_construction(self):
        series = tg.sine_timeseries(length=30, freq="D")
        model = LinearRegressionModel(lags=3, output_chunk_length=OUT_LEN).fit(series)

        # `lam_max` must be positive
        with pytest.raises(ValueError) as exc:
            ConformalDPModel(model=model, quantiles=q, lam_max=0.0)
        assert str(exc.value) == "`lam_max` must be larger than `0`."

        # `lam_init` must be in `[0, lam_max]`
        with pytest.raises(ValueError) as exc:
            ConformalDPModel(model=model, quantiles=q, lam_init=51.0)
        assert str(exc.value) == "`lam_init` must be in `[0, lam_max]` or `None`."

        # `step_size` must be in `(0, 1)`
        with pytest.raises(ValueError) as exc:
            ConformalDPModel(model=model, quantiles=q, step_size=1.0)
        assert str(exc.value) == "`step_size` must be in `(0, 1)`."

        # requires at least one interval
        with pytest.raises(ValueError) as exc:
            ConformalDPModel(model=model, quantiles=[0.5])
        assert (
            str(exc.value)
            == "`quantiles` must contain at least one interval and the median."
        )

        dp_model = ConformalDPModel(model=model, quantiles=q)
        assert dp_model.symmetric
        assert dp_model._residuals_metric[0] is ae

    def test_predict_intervals_valid_and_ordered(self):
        """The DP intervals must be valid quantile output, cover the standard-conformal
        spread for late horizon steps, and tighten where calibration errors are small."""
        series = tg.sine_timeseries(length=60, freq="D", value_y_offset=10.0) + (
            0.1 * tg.gaussian_timeseries(length=60, freq="D")
        )
        # fit on the head, calibrate on the unseen tail (non-zero forecast errors)
        model = train_model(series[:-6])
        dp_model = ConformalDPModel(model=model, quantiles=q)

        pred = dp_model.predict(n=6, series=series, **pred_lklp)
        assert pred.n_components == 3
        assert not np.isnan(pred.all_values()).any().any()

        pred_vals = pred.all_values(copy=False).squeeze(-1)  # (time, comp)
        pred_lo = pred_vals[:, 0]
        pred_med = pred_vals[:, 1]
        pred_hi = pred_vals[:, 2]

        # quantile ordering must be respected
        np.testing.assert_array_less(pred_lo, pred_med)
        np.testing.assert_array_less(pred_med, pred_hi)

        # the median must be the underlying model's forecast
        pred_fc = model.predict(n=6, series=series)
        np.testing.assert_array_almost_equal(pred_med, pred_fc.values().squeeze())

        # average interval width must be comparable to naive conformal (not degenerate)
        naive_model = ConformalNaiveModel(model=train_model(series[:-6]), quantiles=q)
        naive_pred = naive_model.predict(n=6, series=series, **pred_lklp)
        naive_vals = naive_pred.all_values(copy=False).squeeze(-1)
        dp_width = float(np.mean(pred_hi - pred_lo))
        naive_width = float(np.mean(naive_vals[:, 2] - naive_vals[:, 0]))
        assert dp_width > 0.0
        assert dp_width < 3.0 * naive_width

    def test_dp_policy_prefers_narrow_intervals_where_errors_are_small(self):
        """With flat calibration scores at the first steps and a spread at the last, the
        DP policy should allocate coverage budget: not every step takes the widest score."""
        from darts.models.forecasting.conformal_dp_model import (
            solve_bellman_interval_policy,
        )

        rng = np.random.default_rng(42)
        scores = np.sort(np.abs(rng.normal(0.0, 1.0, size=(3, 40))), axis=1)
        # make the last horizon step clearly harder
        scores[2] = np.sort(np.abs(rng.normal(0.0, 3.0, size=40)))

        alpha_bar = 0.2
        ranks = solve_bellman_interval_policy(
            scores=scores, alpha_bar=alpha_bar, lam=2.0, lam_max=50.0
        )
        assert len(ranks) == 3

        # with a positive validity weight, the policy spends the miscoverage budget:
        # not every step takes the widest candidate
        assert not all(rank == len(scores[h]) for h, rank in enumerate(ranks))

        # with the safeguard weight, every step takes the widest candidate
        ranks_safeguard = solve_bellman_interval_policy(
            scores=scores, alpha_bar=alpha_bar, lam=100.0, lam_max=50.0
        )
        assert ranks_safeguard == [len(scores[h]) for h in range(3)]

        # with no validity weight at all, every step takes the narrowest candidate
        ranks_greedy = solve_bellman_interval_policy(
            scores=scores, alpha_bar=alpha_bar, lam=0.0, lam_max=50.0
        )
        assert ranks_greedy == [1, 1, 1]
