import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from darts import TimeSeries
from darts.datasets import AirPassengersDataset
from darts.utils.likelihood_models.sklearn import QuantileRegression
from darts.utils.statistics import (
    check_seasonality,
    extract_trend_and_seasonality,
    granger_causality_tests,
    plot_acf,
    plot_ccf,
    plot_pacf,
    plot_residuals_analysis,
    plot_tolerance_curve,
    remove_from_series,
    remove_seasonality,
    remove_trend,
    rolling_correlations,
    stationarity_test_adf,
    stationarity_test_kpss,
    stationarity_tests,
    temporal_correlation_volatility,
)
from darts.utils.timeseries_generation import (
    constant_timeseries,
    gaussian_timeseries,
    linear_timeseries,
)
from darts.utils.utils import ModelMode, SeasonalityMode

py_312_or_higher = sys.version_info >= (3, 12, 0)


class TestTimeSeries:
    def test_check_seasonality(self):
        pd_series = pd.Series(range(50), index=pd.date_range("20130101", "20130219"))
        pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
        series = TimeSeries.from_series(pd_series)

        assert (True, 6) == check_seasonality(series)
        assert (False, 3) == check_seasonality(series, m=3)

        with pytest.raises(AssertionError):
            check_seasonality(series.stack(series))

    def test_granger_causality(self):
        series_cause_1 = constant_timeseries(start=0, end=9999).stack(
            constant_timeseries(start=0, end=9999)
        )
        series_cause_2 = gaussian_timeseries(start=0, end=9999)
        series_effect_1 = constant_timeseries(start=0, end=999)
        series_effect_2 = TimeSeries.from_values(np.random.uniform(0, 1, 10000))
        series_effect_3 = TimeSeries.from_values(
            np.random.uniform(0, 1, (1000, 2, 1000))
        )
        series_effect_4 = constant_timeseries(
            start=pd.Timestamp("2000-01-01"), length=10000
        )

        # Test univariate
        with pytest.raises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_1, 10)
        with pytest.raises(AssertionError):
            granger_causality_tests(series_effect_1, series_cause_1, 10)

        # Test deterministic
        with pytest.raises(AssertionError):
            granger_causality_tests(series_cause_1, series_effect_3, 10)
        with pytest.raises(AssertionError):
            granger_causality_tests(series_effect_3, series_cause_1, 10)

        # Test Frequency
        with pytest.raises(ValueError):
            granger_causality_tests(series_cause_2, series_effect_4, 10)

        # Test granger basics
        tests = granger_causality_tests(series_effect_2, series_effect_2, 10)
        assert tests[1][0]["ssr_ftest"][1] > 0.99
        tests = granger_causality_tests(series_cause_2, series_effect_2, 10)
        assert tests[1][0]["ssr_ftest"][1] > 0.01

    def test_stationarity_tests(self):
        np.random.seed(42)
        series_1 = constant_timeseries(start=0, end=9999).stack(
            constant_timeseries(start=0, end=9999)
        )

        series_2 = TimeSeries.from_values(np.random.uniform(0, 1, (1000, 2, 1000)))
        series_3 = gaussian_timeseries(start=0, end=9999)

        # Test univariate
        with pytest.raises(AssertionError):
            stationarity_tests(series_1)
        with pytest.raises(AssertionError):
            stationarity_test_adf(series_1)
        with pytest.raises(AssertionError):
            stationarity_test_kpss(series_1)

        # Test deterministic
        with pytest.raises(AssertionError):
            stationarity_tests(series_2)
        with pytest.raises(AssertionError):
            stationarity_test_adf(series_2)
        with pytest.raises(AssertionError):
            stationarity_test_kpss(series_2)

        # Test basics
        assert stationarity_test_kpss(series_3)[1] > 0.05
        assert stationarity_test_adf(series_3)[1] < 0.05
        assert stationarity_tests


class TestSeasonalDecompose:
    pd_series = pd.Series(range(50), index=pd.date_range("20130101", "20130219"))
    pd_series = pd_series.map(lambda x: np.sin(x * np.pi / 3 + np.pi / 2))
    season = TimeSeries.from_series(pd_series)
    trend = linear_timeseries(
        start_value=1, end_value=10, start=season.start_time(), end=season.end_time()
    )
    ts = trend + season

    def test_extract(self):
        series_copy = self.ts.copy()
        # test default (naive) method
        calc_trend, _ = extract_trend_and_seasonality(self.ts, freq=6)
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test default (naive) method additive
        calc_trend, _ = extract_trend_and_seasonality(
            self.ts, freq=6, model=ModelMode.ADDITIVE
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_trend, _ = extract_trend_and_seasonality(
            self.ts, freq=6, method="STL", model=ModelMode.ADDITIVE
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test MSTL method
        calc_trend, calc_seasonality = extract_trend_and_seasonality(
            self.ts, freq=[3, 6], method="MSTL", model=ModelMode.ADDITIVE
        )
        assert len(calc_seasonality.components) == 2
        diff = self.trend - calc_trend
        # relaxed tolerance for MSTL since it will have a larger error from the
        # extrapolation of the trend, it is still a small number but is more
        # than STL or naive trend extraction
        assert np.isclose(np.mean(diff.values() ** 2), 0.0, atol=1e-5)

        # test MSTL method with single freq
        calc_trend, calc_seasonality = extract_trend_and_seasonality(
            self.ts, freq=6, method="MSTL", model=ModelMode.ADDITIVE
        )
        assert len(calc_seasonality.components) == 1
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0, atol=1e-5)

        # make sure non MSTL methods fail with multiple freqs
        with pytest.raises(ValueError):
            calc_trend, calc_seasonality = extract_trend_and_seasonality(
                self.ts, freq=[1, 4, 6], method="STL", model=ModelMode.ADDITIVE
            )

        # check if error is raised when using multiplicative model
        with pytest.raises(ValueError):
            calc_trend, _ = extract_trend_and_seasonality(
                self.ts, freq=6, method="STL", model=ModelMode.MULTIPLICATIVE
            )

        with pytest.raises(ValueError):
            calc_trend, _ = extract_trend_and_seasonality(
                self.ts, freq=[3, 6], method="MSTL", model=ModelMode.MULTIPLICATIVE
            )

        assert self.ts == series_copy

    def test_remove_seasonality(self):
        # test default (naive) method
        calc_trend = remove_seasonality(self.ts, freq=6)
        diff = self.trend - calc_trend
        assert np.mean(diff.values() ** 2).item() < 0.5

        # test default (naive) method additive
        calc_trend = remove_seasonality(self.ts, freq=6, model=SeasonalityMode.ADDITIVE)
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_trend = remove_seasonality(
            self.ts,
            freq=6,
            method="STL",
            model=SeasonalityMode.ADDITIVE,
            low_pass=9,
        )
        diff = self.trend - calc_trend
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # check if error is raised
        with pytest.raises(ValueError):
            calc_trend = remove_seasonality(
                self.ts, freq=6, method="STL", model=SeasonalityMode.MULTIPLICATIVE
            )

    def test_remove_trend(self):
        # test naive method
        calc_season = remove_trend(self.ts, freq=6)
        diff = self.season - calc_season
        assert np.mean(diff.values() ** 2).item() < 1.5

        # test naive method additive
        calc_season = remove_trend(self.ts, freq=6, model=ModelMode.ADDITIVE)
        diff = self.season - calc_season
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # test STL method
        calc_season = remove_trend(
            self.ts,
            freq=6,
            method="STL",
            model=ModelMode.ADDITIVE,
            low_pass=9,
        )
        diff = self.season - calc_season
        assert np.isclose(np.mean(diff.values() ** 2), 0.0)

        # check if error is raised
        with pytest.raises(ValueError):
            calc_season = remove_trend(
                self.ts, freq=6, method="STL", model=ModelMode.MULTIPLICATIVE
            )


class TestPlot:
    series = AirPassengersDataset().load()

    def test_statistics_plot(self, mpl_safe_plotting):
        plot_residuals_analysis(self.series)
        plot_residuals_analysis(self.series, acf_max_lag=10)
        plot_residuals_analysis(self.series[:10])
        plot_acf(self.series)
        plot_pacf(self.series)
        plot_ccf(self.series, self.series)


class TestPlotToleranceCurve:
    # univariate series
    actual_uni = TimeSeries.from_values(np.array([1.0, 2.0, 3.0, 4.0, 5.0]))
    pred_uni = TimeSeries.from_values(np.array([1.1, 2.2, 2.9, 4.1, 5.0]))

    # multivariate series
    actual_multi = TimeSeries.from_values(
        np.column_stack([[1.0, 2.0, 3.0, 4.0, 5.0], [10.0, 20.0, 30.0, 40.0, 50.0]]),
        columns=["c1", "c2"],
    )
    pred_multi = TimeSeries.from_values(
        np.column_stack([[1.1, 2.2, 2.9, 4.1, 5.0], [11.0, 22.0, 29.0, 41.0, 50.0]]),
        columns=["c1", "c2"],
    )

    # multiple multivariate series
    multi_actual_multi = [actual_multi] * 2
    multi_pred_multi = [pred_multi] * 2

    # stochastic series
    pred_stoch = TimeSeries.from_values(
        np.random.rand(5, 1, 10) + np.arange(1.0, 6.0).reshape(-1, 1, 1)
    )
    pred_stoch_multi = TimeSeries.from_values(
        np.random.rand(5, 2, 10) + np.arange(1.0, 6.0).reshape(-1, 1, 1)
    )

    # quantile predictions
    pred_q_uni = TimeSeries.from_values(
        np.random.rand(5, 3, 1),
        columns=QuantileRegression(1, [0.1, 0.5, 0.9]).component_names(actual_uni),
    )
    pred_q_multi = TimeSeries.from_values(
        np.random.rand(5, 6, 1),
        columns=QuantileRegression(1, [0.1, 0.5, 0.9]).component_names(actual_multi),
    )

    @pytest.mark.parametrize(
        "actual,pred,kwargs",
        [
            ("actual_uni", "pred_uni", {}),
            ("actual_multi", "pred_multi", {}),
            ("multi_actual_multi", "multi_pred_multi", {}),
            ("actual_uni", "pred_stoch", {}),
            ("actual_uni", "pred_stoch", {"q": 0.25}),
            ("actual_uni", "pred_stoch", {"q": [0.25, 0.5, 0.75]}),
            ("actual_multi", "pred_stoch_multi", {"q": 0.25}),
            ("actual_multi", "pred_stoch_multi", {"q": [0.25, 0.5, 0.75]}),
            ("actual_uni", "pred_q_uni", {"q": 0.1}),
            ("actual_uni", "pred_q_uni", {"q": 0.5}),
            ("actual_uni", "pred_q_uni", {"q": [0.1, 0.5, 0.9]}),
            ("actual_multi", "pred_q_multi", {"q": 0.1}),
            ("actual_multi", "pred_q_multi", {"q": 0.5}),
            ("actual_multi", "pred_q_multi", {"q": [0.1, 0.5, 0.9]}),
            ("actual_uni", "pred_uni", {"min_tolerance": 0.1, "max_tolerance": 0.9}),
            ("actual_uni", "pred_uni", {"step": 0.05}),
        ],
    )
    def test_plot_tolerance_curve_params(self, mpl_safe_plotting, actual, pred, kwargs):
        plot_tolerance_curve(getattr(self, actual), getattr(self, pred), **kwargs)

    def test_plot_tolerance_curve_with_axis(self, mpl_safe_plotting):
        _, ax = plt.subplots()
        plot_tolerance_curve(self.actual_uni, self.pred_uni, axis=ax)


class TestStatisticsInputValidation:
    ts = constant_timeseries(value=2, length=100)
    ts_other = constant_timeseries(value=1, length=100)
    def test_check_seasonality_invalid_m(self):
        with pytest.raises(ValueError, match="m must be an integer greater than 1"):
            check_seasonality(self.ts, m=1.5)
        with pytest.raises(ValueError, match="m must be an integer greater than 1"):
            check_seasonality(self.ts, m=1)

    def test_check_seasonality_m_exceeds_max_lag(self):
        with pytest.raises(
            ValueError, match="max_lag must be greater than or equal to m"
        ):
            check_seasonality(self.ts, m=10, max_lag=5)

    def test_extract_trend_and_seasonality_invalid_model(self):
        if py_312_or_higher:
            exc = ValueError
            msg = "Unknown value for model_mode"
        else:
            exc = TypeError
            msg = None
        with pytest.raises(exc, match=msg):
            extract_trend_and_seasonality(self.ts, freq=6, model="invalid")

    def test_extract_trend_and_seasonality_none_model(self):
        with pytest.raises(
            ValueError, match="The model must be either MULTIPLICATIVE or ADDITIVE"
        ):
            extract_trend_and_seasonality(self.ts, freq=6, model=SeasonalityMode.NONE)

    def test_extract_trend_and_seasonality_invalid_method(self):
        with pytest.raises(ValueError, match="Unknown value for method"):
            extract_trend_and_seasonality(
                self.ts, freq=6, model=ModelMode.ADDITIVE, method="invalid"
            )

    def test_remove_from_series_invalid_model(self):
        if py_312_or_higher:
            exc = ValueError
            msg = "Unknown value for model_mode"
        else:
            exc = TypeError
            msg = None
        with pytest.raises(exc, match=msg):
            remove_from_series(self.ts, self.ts_other, model="invalid")

    def test_remove_seasonality_none_model(self):
        with pytest.raises(
            ValueError, match="The model must be either MULTIPLICATIVE or ADDITIVE"
        ):
            remove_seasonality(self.ts, freq=6, model=SeasonalityMode.NONE)

    @pytest.mark.parametrize(
        "plot_fn,extra_kwargs",
        [
            (plot_acf, {}),
            (plot_pacf, {}),
            (plot_ccf, {}),
        ],
    )
    def test_plot_invalid_max_lag(self, mpl_safe_plotting, plot_fn, extra_kwargs):
        args = [self.ts]
        if plot_fn is plot_ccf:
            args.append(self.ts)
        with pytest.raises(ValueError, match="max_lag must be greater than or equal"):
            plot_fn(*args, max_lag=0, **extra_kwargs)
        with pytest.raises(ValueError, match="max_lag must be greater than or equal"):
            plot_fn(*args, max_lag=len(self.ts), **extra_kwargs)

    @pytest.mark.parametrize(
        "plot_fn",
        [plot_acf, plot_pacf, plot_ccf],
    )
    def test_plot_invalid_m(self, mpl_safe_plotting, plot_fn):
        args = [self.ts]
        if plot_fn is plot_ccf:
            args.append(self.ts)
        with pytest.raises(
            ValueError,
            match="m must be greater than or equal to 0 and less than or equal to max_lag",
        ):
            plot_fn(*args, max_lag=10, m=11)

    @pytest.mark.parametrize(
        "plot_fn",
        [plot_acf, plot_pacf, plot_ccf],
    )
    def test_plot_invalid_alpha(self, mpl_safe_plotting, plot_fn):
        args = [self.ts]
        if plot_fn is plot_ccf:
            args.append(self.ts)
        with pytest.raises(
            ValueError, match="alpha must be greater than 0 and less than 1"
        ):
            plot_fn(*args, max_lag=10, alpha=0)
        with pytest.raises(
            ValueError, match="alpha must be greater than 0 and less than 1"
        ):
            plot_fn(*args, max_lag=10, alpha=1)


class TestCorrelationVolatility:
    n_steps = 400
    # component 2 flips from a perfect copy of component 1 to its mirror image
    z = np.random.default_rng(42).normal(size=n_steps)
    sign = np.where(np.arange(n_steps) < n_steps // 2, 1.0, -1.0)
    ts_dynamic = TimeSeries.from_values(np.stack([z, z * sign], axis=1))
    # both components stay noisy copies of the same driver
    ts_static = TimeSeries.from_values(
        np.stack([z, z + 0.1 * np.random.default_rng(43).normal(size=n_steps)], axis=1)
    )

    def test_tcv_detects_changing_correlations(self):
        # a re-wiring structure must score clearly above a stable one
        assert (
            temporal_correlation_volatility(self.ts_dynamic, window=50)
            > 10 * temporal_correlation_volatility(self.ts_static, window=50)
        )

    def test_tcv_static_structure_is_close_to_zero(self):
        assert temporal_correlation_volatility(self.ts_static, window=50) < 0.01

    def test_tcv_is_bounded(self):
        tcv = temporal_correlation_volatility(self.ts_dynamic, window=50)
        assert 0.0 < tcv <= 2.0

    def test_rolling_correlations_recover_the_flip(self):
        correlations = rolling_correlations(self.ts_dynamic, window=50)
        assert correlations.width == 4  # flattened 2x2 matrix
        assert len(correlations) == self.n_steps - 50 + 1
        # entry "0~1" is the off-diagonal: starts at +1, ends at -1
        off_diagonal = correlations.values(copy=False)[:, 1]
        assert off_diagonal[0] == pytest.approx(1.0, abs=1e-6)
        assert off_diagonal[-1] == pytest.approx(-1.0, abs=1e-6)

    def test_rolling_correlations_keeps_time_index(self):
        ts = self.ts_static.with_times_and_values(
            times=pd.date_range("2020-01-01", periods=self.n_steps, freq="D"),
            values=self.ts_static.values(copy=False),
        )
        correlations = rolling_correlations(ts, window=50)
        assert isinstance(correlations.time_index, pd.DatetimeIndex)
        assert correlations.time_index[0] == pd.Timestamp("2020-02-19")
        assert correlations.time_index[-1] == ts.time_index[-1]

    def test_tcv_stride_and_min_periods(self):
        # a stride larger than the flip boundary still sees the re-wiring
        with_stride = temporal_correlation_volatility(
            self.ts_dynamic, window=50, stride=10
        )
        assert with_stride > 0
        # windows containing NaNs are dropped unless min_periods is lowered
        values = self.ts_static.values(copy=True)
        values[:3, 1] = np.nan
        ts_with_nans = TimeSeries.from_values(values)
        assert temporal_correlation_volatility(ts_with_nans, window=50) > 0.0
        with_min_periods = temporal_correlation_volatility(
            ts_with_nans, window=50, min_periods=10
        )
        assert with_min_periods > 0.0


class TestCorrelationVolatilityInputValidation:
    ts_multi = TimeSeries.from_values(np.random.default_rng(0).normal(size=(20, 2)))
    ts_univariate = TimeSeries.from_values(np.arange(20).astype(float))

    def test_invalid_window(self):
        with pytest.raises(ValueError, match="window must be between 2"):
            temporal_correlation_volatility(self.ts_multi, window=1)
        with pytest.raises(ValueError, match="window must be between 2"):
            temporal_correlation_volatility(self.ts_multi, window=21)

    def test_invalid_stride(self):
        with pytest.raises(ValueError, match="stride must be at least 1"):
            temporal_correlation_volatility(self.ts_multi, window=10, stride=0)

    def test_univariate_series(self):
        with pytest.raises(ValueError, match="only defined for multivariate series"):
            temporal_correlation_volatility(self.ts_univariate, window=10)
        with pytest.raises(ValueError, match="only defined for multivariate series"):
            rolling_correlations(self.ts_univariate, window=10)

    def test_too_few_windows(self):
        with pytest.raises(ValueError, match="fewer than two windows"):
            temporal_correlation_volatility(self.ts_multi, window=20)
