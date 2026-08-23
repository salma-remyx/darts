"""
Label Shift Weighting for Conformal Calibration
------------------------------------------------

Density-ratio weights that re-weight a calibration set so that it matches the
label (target value) distribution of the period being predicted.

Under label shift the calibration residuals are drawn from a different target
distribution than the one the intervals must cover, which biases the standard
exchangeable conformal quantile. Following the covariate-shift treatment of
Tibshirani et al. (2019), weighting each non-conformity score by the ratio of
the target density in the evaluation window over the target density in the
calibration window restores approximate validity.

Adapted from "Conformal Prediction for Molecular Properties under Label Shift"
(arXiv:2608.17678), which weights conformal scores by marginal label
probability ratios so that prediction intervals stay calibrated when the
property distribution drifts. The molecular framing is the motivating
application only; the ratio-of-marginals weighting is domain-agnostic and is
applied here to the value axis of darts forecasting series.

The marginals are estimated with equal-width histograms (a parameter-free
plug-in estimator of the label density), and the ratios are clipped to keep
the weights finite and bounded when a bin is empty in either window.
"""

import numpy as np

DEFAULT_LABEL_SHIFT_BINS = 20
DEFAULT_LABEL_SHIFT_CLIP = 10.0


def _hist_density(
    values: np.ndarray, edges: np.ndarray, n_samples: int
) -> np.ndarray:
    """Histogram density over `edges` for a 1-d array of label values."""
    counts, _ = np.histogram(values, bins=edges)
    width = np.diff(edges)
    # `n_samples` is the total number of labels of the window (including the
    # ones outside `edges`), so the densities integrate to <= 1.
    return counts / (n_samples * width)


def _shared_edges(
    values: np.ndarray, other: np.ndarray, n_bins: int
) -> np.ndarray:
    """Bin edges spanning both windows, so the two histograms are comparable."""
    all_values = np.concatenate([values, other])
    lo = np.nanmin(all_values)
    hi = np.nanmax(all_values)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # degenerate range (constant labels): a single bin, all weights are 1
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins + 1)


def label_shift_weights(
    cal_labels: np.ndarray,
    eval_labels: np.ndarray,
    n_bins: int = DEFAULT_LABEL_SHIFT_BINS,
    clip: float = DEFAULT_LABEL_SHIFT_CLIP,
) -> np.ndarray:
    """Estimate label-shift weights for a calibration set.

    Each calibration label is weighted by the ratio of the target value
    density in the evaluation window over the target value density in the
    calibration window, both estimated with equal-width histograms over a
    shared bin grid. Labels of the evaluation window that fall outside the
    calibration range get their nearest-edge bin, so a drifting window never
    yields a zero denominator.

    Parameters
    ----------
    cal_labels
        1-d array of target values observed in the calibration window.
    eval_labels
        1-d array of target values observed in the evaluation (recent) window.
        Can be empty, in which case no shift is detectable and all weights
        are `1`.
    n_bins
        Number of equal-width histogram bins used to estimate both marginals.
    clip
        Upper bound on the density ratio. Clipping keeps the weighted quantile
        well-behaved when a bin is densely populated in the evaluation window
        but nearly empty in the calibration window.

    Returns
    -------
    np.ndarray
        Weights of shape `(len(cal_labels),)`, strictly positive.
    """
    cal_labels = np.asarray(cal_labels, dtype=float).reshape(-1)
    eval_labels = np.asarray(eval_labels, dtype=float).reshape(-1)

    if cal_labels.size == 0:
        return np.empty(0)

    if eval_labels.size == 0:
        # no information about the target distribution: exchangeable weights
        return np.ones_like(cal_labels)

    edges = _shared_edges(cal_labels, eval_labels, n_bins)

    cal_density = _hist_density(cal_labels, edges, cal_labels.size)
    eval_density = _hist_density(eval_labels, edges, eval_labels.size)

    # bin index of every calibration label; out-of-range values are clipped to
    # the nearest edge bin so the ratio stays defined
    cal_idx = np.clip(
        np.searchsorted(edges, cal_labels, side="right") - 1, 0, n_bins - 1
    )

    # add-one smoothing on the calibration marginal avoids division by zero and
    # keeps every weight finite
    cal_counts, _ = np.histogram(cal_labels, bins=edges)
    cal_density_s = cal_density + (1.0 / cal_labels.size) * (cal_density > 0)
    cal_density_s = np.where(cal_counts == 0, 1.0 / cal_labels.size, cal_density_s)

    ratio = eval_density[cal_idx] / cal_density_s[cal_idx]
    return np.clip(ratio, 1.0 / clip, clip)


def weighted_quantile(
    values: np.ndarray, q: float | list[float] | np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """Quantile(s) of `values` under non-negative sample `weights`.

    Inverse of the weighted empirical CDF. Ties in `values` are aggregated by
    summing their weights, which makes the result independent of the input
    ordering.

    Returns
    -------
    np.ndarray
        Array of quantile values, one per entry of `q`.
    """
    values = np.asarray(values, dtype=float).reshape(-1)
    weights = np.asarray(weights, dtype=float).reshape(-1)
    if values.size == 0:
        raise ValueError("`values` must be non-empty.")
    if values.shape != weights.shape:
        raise ValueError("`values` and `weights` must have the same length.")
    if np.any(weights < 0):
        raise ValueError("`weights` must be non-negative.")

    if np.all(weights == 0):
        # uninformative weights: fall back to the unweighted empirical quantile
        weights = np.ones_like(weights)

    qs = np.atleast_1d(np.asarray(q, dtype=float))

    order = np.argsort(values, kind="stable")
    values_s = values[order]
    weights_s = weights[order]

    # aggregate ties so that the weighted CDF is a step function of the values
    values_u, inverse = np.unique(values_s, return_inverse=True)
    weights_u = np.zeros(values_u.size)
    np.add.at(weights_u, inverse, weights_s)

    cdf = np.cumsum(weights_u) / weights_u.sum()
    # conservative inverse CDF (matches `np.quantile(..., method="higher")` for uniform
    # weights): first value whose cumulative weight strictly exceeds `q`
    idx = np.searchsorted(cdf, qs, side="right")
    idx = np.clip(idx, 0, values_u.size - 1)
    return values_u[idx]
