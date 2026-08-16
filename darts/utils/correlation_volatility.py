"""
Correlation Volatility
----------------------

Diagnostics quantifying how much the pairwise correlation structure of a
multivariate time series drifts over time.
"""

import numpy as np
import pandas as pd

from darts import TimeSeries
from darts.logging import get_logger, raise_log
from darts.utils.ts_utils import get_single_series

logger = get_logger(__name__)


def _pairwise_corrcoef(x: np.ndarray, pairs) -> np.ndarray:
    """
    Computes the Pearson correlation of every component pair of `x`, each on its
    pairwise-complete observations (the rows where both components are observed).
    """
    return np.array(
        [
            np.corrcoef(x[valid, col_i], x[valid, col_j])[0, 1]
            for col_i, col_j, valid in pairs
        ]
    )


def _iter_pairs(x: np.ndarray, i: np.ndarray, j: np.ndarray):
    """
    Yields ``(column_i, column_j, valid_row_indices)`` for every component pair, where the
    valid rows are those where both components of the pair are observed.
    """
    observed = ~np.isnan(x)
    for col_i, col_j in zip(i, j):
        yield col_i, col_j, np.flatnonzero(observed[:, col_i] & observed[:, col_j])


def _windowed_correlations(
    series: TimeSeries,
    window: int,
    stride: int,
    min_periods: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the upper-triangle Pearson correlations of `series` on consecutive rolling windows.

    Parameters
    ----------
    series
        The multivariate time series. Its components are the variables being correlated.
    window
        Length of the rolling window (number of time steps). Must be at least 2 and at
        most the length of the series.
    stride
        Step between the start of two consecutive windows. Must be at least 1.
    min_periods
        Minimum number of valid (non-NaN) observations a window must contain to be kept.
        Defaults to ``window``, i.e. windows with any missing value are dropped.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        A tuple `(values, starts)` where `values` has one row of upper-triangle (diagonal
        excluded) correlation coefficients per window, and `starts` holds the start index
        of each retained window.
    """
    n_steps = len(series)
    if not 2 <= window <= n_steps:
        raise_log(
            ValueError(
                f"window must be between 2 and the number of time steps ({n_steps}), "
                f"received {window}."
            )
        )
    if stride < 1:
        raise_log(ValueError(f"stride must be at least 1, received {stride}."))
    if min_periods is None:
        min_periods = window
    if not 2 <= min_periods <= window:
        raise_log(
            ValueError(
                f"min_periods must be between 2 and window ({window}), "
                f"received {min_periods}."
            )
        )

    n_components = series.width
    if n_components < 2:
        raise_log(
            ValueError(
                "correlation volatility is only defined for multivariate series with at "
                f"least 2 components, received {n_components}."
            )
        )

    x = series.values(copy=False)
    if x.ndim == 3:  # stochastic series: squeeze the sample dimension
        x = x[:, :, 0]

    i, j = np.triu_indices(n_components, k=1)
    rows = []
    starts = []
    for start in range(0, n_steps - window + 1, stride):
        block = x[start : start + window]
        # correlations are computed on pairwise-complete observations, i.e. the rows
        # where both components of a pair are observed
        pairs = list(_iter_pairs(block, i, j))
        if any(len(valid) < min_periods for _, _, valid in pairs):
            continue
        rows.append(_pairwise_corrcoef(block, pairs))
        starts.append(start)
    if len(rows) < 2:
        raise_log(
            ValueError(
                f"fewer than two windows had at least min_periods={min_periods} valid "
                "observations, so the correlation structure cannot be compared over "
                "time; lower min_periods, decrease stride, or decrease window."
            )
        )

    return np.stack(rows), np.asarray(starts)


def temporal_correlation_volatility(
    series: TimeSeries,
    window: int,
    stride: int = 1,
    min_periods: int | None = None,
) -> float:
    """
    Computes the Temporal Correlation Volatility (TCV) of a multivariate time series.

    TCV quantifies the distributional evolution of the latent correlation structure: the
    series is split into rolling windows, the pairwise Pearson correlation between
    components is computed within each window, and TCV is the mean absolute change of
    those correlations between consecutive windows. It is model-agnostic and is a proxy
    for how much the "graph" linking the series (one node per component, one edge per
    pairwise correlation) re-wires over time.

    High TCV indicates that the relationships between components are unstable, a regime in
    which models relying on a fixed dependency structure (e.g. transformers, graph neural
    networks) tend to be outperformed by structure-agnostic baselines. A perfectly rigid
    structure gives a TCV of ``0.0``. At least two windows must be retained, i.e.
    ``window + stride`` must not exceed the number of time steps.

    Adapted from "When GNNs Fail: Quantifying and Overcoming Temporal Correlation
    Volatility in Time Series" (https://arxiv.org/abs/2608.07333), which introduces TCV
    and relates it to forecasting performance degradation.

    Parameters
    ----------
    series
        The multivariate time series; each component is one node of the correlation graph.
    window
        Length of the rolling window (number of time steps) over which correlations are
        computed. Shorter windows detect faster re-wiring but give noisier estimates; a
        common choice is the forecasting model's expected lookback length.
    stride
        Step between the start of two consecutive windows (default 1, i.e. fully overlapping
        windows). Larger strides give sparser, cheaper estimates.
    min_periods
        Minimum number of valid (non-NaN) observations a window must contain to be kept.
        Defaults to ``window``, i.e. windows with any missing value are dropped.

    Returns
    -------
    float
        The TCV score: the mean absolute difference between the upper triangles of
        consecutive window correlation matrices, bounded by ``[0, 2]`` since each
        correlation coefficient lies in ``[-1, 1]``.

    Examples
    --------
    >>> import numpy as np
    >>> from darts import TimeSeries
    >>> z = np.random.default_rng(0).normal(size=500)
    >>> # components flip from perfectly correlated to perfectly anti-correlated
    >>> values = np.stack([z, np.where(np.arange(500) < 250, z, -z)], axis=1)
    >>> series = TimeSeries.from_values(values)
    >>> temporal_correlation_volatility(series, window=50) > 0.0
    True
    """
    values, _ = _windowed_correlations(
        get_single_series(series), window, stride, min_periods
    )
    return float(np.mean(np.abs(np.diff(values, axis=0))))


def rolling_correlations(
    series: TimeSeries,
    window: int,
    stride: int = 1,
    min_periods: int | None = None,
) -> TimeSeries:
    """
    Computes the rolling Pearson correlation matrices of a multivariate time series.

    This is the windowed decomposition that :func:`temporal_correlation_volatility`
    aggregates; it is exposed so the correlation dynamics can be inspected, e.g. to see
    which component pairs drive a high volatility score.

    Parameters
    ----------
    series
        The multivariate time series.
    window
        Length of the rolling window (number of time steps).
    stride
        Step between the start of two consecutive windows (default 1).
    min_periods
        Minimum number of valid (non-NaN) observations a window must contain to be kept.
        Defaults to ``window``, i.e. windows with any missing value are dropped.

    Returns
    -------
    TimeSeries
        A deterministic series with one component per matrix entry, named
        ``"<component_i>~<component_j>"``, and one time stamp per window (the last step of
        the window). Reshaping a row to ``(n_components, n_components)`` recovers the full
        symmetric correlation matrix of that window.

    Examples
    --------
    >>> import numpy as np
    >>> from darts import TimeSeries
    >>> series = TimeSeries.from_values(np.arange(40).reshape(20, 2))
    >>> rolling_correlations(series, window=10).width
    4
    """
    single = get_single_series(series)
    values, starts = _windowed_correlations(single, window, stride, min_periods)
    end_time_idx = starts + window - 1
    if single.has_range_index:
        time_index = pd.RangeIndex(
            start=int(end_time_idx[0]),
            stop=int(end_time_idx[-1]) + 1,
            step=int(end_time_idx[1] - end_time_idx[0]),
        )
    else:
        time_index = single.time_index[end_time_idx]

    n_components = single.width
    i, j = np.triu_indices(n_components, k=1)
    rows = []
    for corr_row in values:
        corr = np.eye(n_components)
        corr[i, j] = corr_row
        corr[j, i] = corr_row
        rows.append(corr.reshape(-1))
    columns = single.columns
    components = [f"{name_i}~{name_j}" for name_i in columns for name_j in columns]
    return TimeSeries.from_times_and_values(
        times=time_index,
        values=np.stack(rows).astype(single.dtype),
        columns=components,
    )
