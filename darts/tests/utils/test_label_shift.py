import numpy as np
import pytest

from darts import TimeSeries
from darts.models import ConformalNaiveModel, LinearRegressionModel
from darts.utils.label_shift import label_shift_weights, weighted_quantile

q = [0.1, 0.5, 0.9]


class TestLabelShiftWeights:
    def test_no_shift_gives_uniform_weights(self):
        """Without label shift the calibration and evaluation marginals are the same
        population, so the density ratios average out to one."""
        rng = np.random.default_rng(42)
        labels = rng.normal(0.0, 1.0, 5000)
        weights = label_shift_weights(labels, labels.copy(), n_bins=20)
        assert weights.shape == labels.shape
        np.testing.assert_allclose(weights.mean(), 1.0, rtol=0.15)
        assert np.all(weights > 0)

    def test_empty_evaluation_window(self):
        """With no evaluation labels there is no observable shift: uniform weights."""
        rng = np.random.default_rng(42)
        labels = rng.normal(0.0, 1.0, 50)
        np.testing.assert_array_equal(
            label_shift_weights(labels, np.empty(0), n_bins=10), np.ones(50)
        )
        np.testing.assert_array_equal(
            label_shift_weights(np.empty(0), labels, n_bins=10), np.empty(0)
        )

    def test_shift_upweights_matching_labels(self):
        """When the evaluation window sits at higher values than the calibration window,
        the high-valued calibration labels must get the larger weights."""
        cal = np.linspace(0.0, 1.0, 100)
        ev = np.linspace(1.0, 2.0, 50)
        weights = label_shift_weights(cal, ev, n_bins=10)
        # the top half of the calibration range should dominate the bottom half
        assert weights[50:].sum() > weights[:50].sum()
        # weights stay strictly positive and bounded
        assert np.all(weights > 0)
        assert np.all(weights <= 10.0 + 1e-9)

    def test_weights_are_finite_for_disjoint_windows(self):
        """Completely disjoint windows must not produce inf / nan weights."""
        cal = np.zeros(30)
        ev = np.full(30, 5.0)
        weights = label_shift_weights(cal, ev, n_bins=10)
        assert np.all(np.isfinite(weights))

    def test_constant_labels(self):
        """A degenerate (constant) label range must not crash."""
        weights = label_shift_weights(np.ones(20), np.ones(20), n_bins=5)
        assert weights.shape == (20,)
        assert np.all(np.isfinite(weights))


class TestWeightedQuantile:
    def test_matches_numpy_for_uniform_weights(self):
        rng = np.random.default_rng(7)
        values = rng.normal(0.0, 1.0, 200)
        weights = np.full_like(values, 2.5)  # constant weights == unweighted
        for q_ in [0.1, 0.5, 0.9]:
            np.testing.assert_allclose(
                weighted_quantile(values, q_, weights),
                np.quantile(values, q_, method="higher"),
            )

    def test_upweighting_pulls_the_quantile_towards_a_subpopulation(self):
        values = np.concatenate([np.zeros(50), np.ones(50)])
        q_low = weighted_quantile(
            values, 0.5, np.concatenate([np.ones(50), np.zeros(50)])
        )
        q_high = weighted_quantile(
            values, 0.5, np.concatenate([np.zeros(50), np.ones(50)])
        )
        assert q_low == 0.0
        assert q_high == 1.0

    def test_invalid_inputs(self):
        with pytest.raises(ValueError):
            weighted_quantile(np.empty(0), 0.5, np.empty(0))
        with pytest.raises(ValueError):
            weighted_quantile(np.ones(3), 0.5, np.ones(4))
        with pytest.raises(ValueError):
            weighted_quantile(np.ones(3), 0.5, np.full(3, -1.0))

    def test_all_zero_weights_fall_back_to_unweighted(self):
        values = np.arange(5.0)
        np.testing.assert_allclose(
            weighted_quantile(values, 0.5, np.zeros(5)), np.quantile(values, 0.5)
        )

    def test_order_independent(self):
        values = np.array([3.0, 1.0, 2.0, 5.0, 4.0])
        weights = np.array([1.0, 2.0, 0.5, 1.5, 1.0])
        perm = np.array([4, 2, 0, 3, 1])
        np.testing.assert_allclose(
            weighted_quantile(values, [0.25, 0.5, 0.75], weights),
            weighted_quantile(values[perm], [0.25, 0.5, 0.75], weights[perm]),
        )


class TestConformalLabelShiftIntegration:
    """The `label_shift_bins` parameter of the conformal models must be wired through
    `predict()` / `historical_forecasts()` all the way to the calibration quantiles."""

    @staticmethod
    def _shifted_series():
        """A series whose target distribution shifts upwards half-way through."""
        rng = np.random.default_rng(0)
        values = np.concatenate([
            rng.normal(0.0, 0.3, 80),
            rng.normal(2.0, 0.6, 60),
        ])
        return TimeSeries.from_values(values)

    def test_predict_changes_intervals_under_shift(self):
        series = self._shifted_series()
        model_fc = LinearRegressionModel(lags=3, output_chunk_length=3).fit(series)

        pred_off = ConformalNaiveModel(model=model_fc, quantiles=q).predict(
            3, series=series, num_samples=1, predict_likelihood_parameters=True
        )
        pred_on = ConformalNaiveModel(
            model=model_fc, quantiles=q, label_shift_bins=20
        ).predict(3, series=series, num_samples=1, predict_likelihood_parameters=True)

        # the median forecast is the underlying model's, untouched by the adaptation
        np.testing.assert_allclose(
            pred_on.values()[:, 1], pred_off.values()[:, 1], rtol=1e-12
        )
        # the interval bounds are re-calibrated under the drifted label distribution
        assert not np.allclose(pred_on.values(), pred_off.values())

    def test_historical_forecasts_with_label_shift(self):
        series = self._shifted_series()
        model_fc = LinearRegressionModel(lags=3, output_chunk_length=3).fit(series)
        model = ConformalNaiveModel(
            model=model_fc, quantiles=q, label_shift_bins=10, label_shift_eval_length=10
        )
        for last_points_only in [True, False]:
            preds = model.historical_forecasts(
                series,
                forecast_horizon=3,
                stride=1,
                last_points_only=last_points_only,
                predict_likelihood_parameters=True,
                num_samples=1,
                retrain=False,
            )
            preds = preds if last_points_only else preds[0]
            # quantile ordering must be preserved: lower bound <= median <= upper bound
            values = preds.values() if last_points_only else preds[-1].values()
            assert np.all(values[:, 0] <= values[:, 1] + 1e-12)
            assert np.all(values[:, 1] <= values[:, 2] + 1e-12)

    def test_disabled_by_default(self):
        series = self._shifted_series()
        model_fc = LinearRegressionModel(lags=3, output_chunk_length=3).fit(series)
        model = ConformalNaiveModel(model=model_fc, quantiles=q)
        assert model.label_shift_bins is None
        pred = model.predict(
            3, series=series, num_samples=1, predict_likelihood_parameters=True
        )
        assert pred.values().shape == (3, 3)

    def test_parameter_validation(self):
        model_fc = LinearRegressionModel(lags=3, output_chunk_length=3).fit(
            TimeSeries.from_values(np.arange(20, dtype=float))
        )
        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=model_fc, quantiles=q, label_shift_bins=1)
        assert str(exc.value) == "`label_shift_bins` must be `>=2` or `None`."

        with pytest.raises(ValueError) as exc:
            ConformalNaiveModel(model=model_fc, quantiles=q, label_shift_eval_length=0)
        assert str(exc.value) == "`label_shift_eval_length` must be `>=1`."

    def test_coverage_improves_under_shift(self):
        """End-to-end check on a drifting series: the adapted intervals must not
        undercover the held-out labels more than the unadapted ones."""
        rng = np.random.default_rng(1)
        values = np.concatenate([
            rng.normal(0.0, 0.4, 120),
            rng.normal(3.0, 0.8, 80),
        ])
        series = TimeSeries.from_values(values)
        n = 3
        model_fc = LinearRegressionModel(lags=3, output_chunk_length=3).fit(series)

        coverages = {}
        for label_shift_bins in [None, 20]:
            model = ConformalNaiveModel(
                model=model_fc, quantiles=q, label_shift_bins=label_shift_bins
            )
            preds = model.historical_forecasts(
                series,
                forecast_horizon=n,
                stride=1,
                last_points_only=True,
                predict_likelihood_parameters=True,
                num_samples=1,
                retrain=False,
            )
            actuals = series.values(copy=False)[-(len(preds) - 1) :][: len(preds) - 1]
            preds_vals = preds.values()[:-1]
            covered = (actuals <= preds_vals[:, 2]) & (actuals >= preds_vals[:, 0])
            coverages[label_shift_bins] = covered.mean()

        # the 80% interval should be reached, and the adapted one must not do worse
        assert coverages[20] >= coverages[None]
