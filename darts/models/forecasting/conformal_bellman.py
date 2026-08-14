"""
Bellman Conformal Prediction Model
----------------------------------

Conformal prediction model that optimizes average interval lengths across a
multi-step horizon by solving a one-dimensional stochastic control problem.

Adapted from "Bellman Conformal Inference: Calibrating Prediction Intervals
For Time Series" (Yang, Candès & Lei, 2024, arXiv:2402.05203).
"""

import numpy as np

from darts.logging import raise_log
from darts.models.forecasting.conformal_models import ConformalNaiveModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel

# number of grid points used to discretize the miscoverage level `[0, 1]` in the
# dynamic program; the reference implementation refers to this as `bins`
_N_ALPHA_BINS = 101


def bellman_interval_widths(
    scores: np.ndarray,
    alpha0: float,
    lam: float | None = None,
    n_bins: int = _N_ALPHA_BINS,
) -> np.ndarray:
    """Optimize per-step interval half-widths against a shared miscoverage budget.

    Solves the horizon-coupled stochastic control problem from Bellman Conformal
    Inference by backward induction over the cumulative-miscoverage state. Each
    horizon step `h` chooses a miscoverage level `alpha_h` such that the sum of
    `alpha_h` over the horizon respects the long-run miscoverage budget
    ``alpha0 * horizon`` while the total interval length is minimized:

    .. math::
        \\min_{\\alpha_{1..H}} \\sum_{h=1}^{H} w_h(\\alpha_h)
        \\quad \\text{s.t.} \\quad \\sum_{h=1}^{H} \\alpha_h \\leq \\alpha_0 H

    where ``w_h(alpha)`` is the interval half-width at horizon step `h` obtained
    as the ``1 - alpha`` quantile of the calibration non-conformity scores at
    that step. A single step's width cannot be minimized in isolation: spending
    more budget where it buys little width leaves less for steps where it buys
    a lot, which is exactly the coupling the Bellman recursion resolves.

    Parameters
    ----------
    scores
        Non-conformity scores with shape ``(horizon, n calibration examples)``.
    alpha0
        Target long-run miscoverage level, e.g. ``0.2`` for 80% intervals.
    lam
        Terminal cost weight of the dynamic program (the paper's ``lambda``).
        Smaller values prioritize short intervals, larger values prioritize
        meeting the miscoverage budget. If `None` (default), `lam` is found by
        bisection so that the budget constraint is met with equality.
    n_bins
        Discretization accuracy of the miscoverage level ``[0, 1]``.

    Returns
    -------
    np.ndarray
        Interval half-widths with shape ``(horizon,)``.
    """
    horizon, n_cal = scores.shape
    if not 0.0 < alpha0 < 1.0:
        raise_log(ValueError("`alpha0` must be in `(0, 1)`."))
    if n_cal < 1:
        raise_log(ValueError("`scores` must contain at least one example."))

    # candidate miscoverage levels; the minimum avoids an empty interval at the
    # smallest level (the "infinite interval" degenerate case)
    alphas = np.linspace(1.0 / (n_cal + 1.0), 1.0, n_bins)
    # half-width of the symmetric interval at each (step, alpha): the
    # `1 - alpha` quantile of the calibration scores, shape (horizon, n_bins)
    widths = np.quantile(scores, 1.0 - alphas, axis=1, method="higher").T

    states = np.arange(horizon + 1)

    def solve_for(lam_):
        """Backward induction: returns the marginal costs used by the forward pass."""
        # terminal cost: weighted penalty on exceeding the budget
        cost_to_go = lam_ * np.maximum(states / horizon - alpha0, 0.0)
        marginal = np.empty((horizon, horizon + 1))
        for h in range(horizon - 1, -1, -1):
            # marginal cost of one additional miss in each state; pad the last
            # state with its neighbor since J is only defined on `states`
            diff = np.maximum(np.diff(cost_to_go), 0.0)
            marginal[h] = np.append(diff, diff[-1])
            # Bellman update: expected cost of choosing each alpha, minimized
            # over alpha, added to the current cost-to-go
            cost_to_go = cost_to_go + np.min(
                widths[h][None, :]
                + alphas[None, :] * marginal[h][:, None],
                axis=1,
            )
        # forward pass: walk the horizon greedily using the marginal costs,
        # tracking the expected cumulative miscoverage state
        rho = 0.0
        alpha_path = np.empty(horizon)
        for h in range(horizon):
            d = float(np.interp(rho, states, marginal[h]))
            alpha_path[h] = alphas[np.argmin(widths[h] + alphas * d)]
            rho += alpha_path[h]
        widths_path = np.array([
            np.interp(alpha_path[h], alphas, widths[h]) for h in range(horizon)
        ])
        return alpha_path, widths_path, rho

    if lam is not None:
        _, widths_path, _ = solve_for(lam)
        return widths_path

    # find `lam` by bisection such that the total expected miscoverage meets
    # the budget `alpha0 * horizon`
    budget = alpha0 * horizon
    lo, hi = 0.0, 1.0
    for _ in range(60):
        if solve_for(hi)[2] <= budget:
            break
        hi *= 4.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        if solve_for(mid)[2] <= budget:
            hi = mid
        else:
            lo = mid
    _, widths_path, _ = solve_for(hi)
    return widths_path


class ConformalBellmanModel(ConformalNaiveModel):
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: list[float],
        symmetric: bool = True,
        cal_length: int | None = None,
        cal_stride: int = 1,
        cal_num_samples: int = 500,
        lam: float | None = None,
        random_state: int | None = None,
    ):
        """Bellman Conformal Prediction Model.

        A probabilistic model that calibrates the median forecast of any
        pre-trained global forecasting model, choosing per-step interval widths
        that jointly minimize the total interval length across the multi-step
        horizon while meeting the target long-run coverage.

        Standard conformal models (including :class:`ConformalNaiveModel`)
        calibrate each horizon step independently at a fixed quantile level, so
        every step is forced to carry the same coverage regardless of how
        wide the interval is there. This model instead treats the horizon as a
        one-dimensional stochastic control problem (Yang, Candès & Lei, 2024):
        a shared miscoverage budget ``alpha0 * horizon`` is allocated across
        steps by dynamic programming, which can spend more budget at steps
        where additional coverage is expensive in width, and less where it is
        cheap — producing shorter intervals on average at the same overall
        coverage.

        Non-conformity scores are the same as :class:`ConformalNaiveModel`:
        absolute errors of the median forecast (`symmetric=True`) or signed
        errors for the lower/upper bounds separately (`symmetric=False`).

        Since it is a probabilistic model, you can generate forecasts in two
        ways (when calling `predict()`, `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters
          `predict_likelihood_parameters=True`, and `num_samples=1` to the
          forecast method.
        - Predict stochastic samples from the calibrated quantile intervals:
          Pass parameters `predict_likelihood_parameters=False`, and
          `num_samples>>1` to the forecast method.

        Parameters
        ----------
        model
            A pre-trained global forecasting model. See the list of models
            `here <https://unit8co.github.io/darts/#forecasting-models>`__.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. The
            innermost interval's coverage determines the target long-run
            miscoverage level ``alpha0 = 1 - (q_highest - q_lowest)`` used by
            the dynamic program, e.g. quantiles [0.1, 0.5, 0.9] give
            ``alpha0 = 0.2``.
        symmetric
            Whether to use symmetric non-conformity scores. If `True`, uses
            metric `ae()` (see :func:`~darts.metrics.metrics.ae`). If `False`,
            uses metric `-err()` / `err()` for the lower/upper bounds (see
            :func:`~darts.metrics.metrics.err`).
        cal_length
            The number of past forecast errors / non-conformity scores to use
            as calibration for each conformal forecast (and each step in the
            horizon). If `None`, considers all scores.
        cal_stride
            The stride to apply when computing the historical forecasts and
            non-conformity scores on the calibration set.
        cal_num_samples
            The number of samples to generate for each calibration forecast (if
            `model` is a probabilistic forecasting model).
        lam
            Terminal cost weight of the dynamic program (the paper's
            ``lambda``). If `None` (default), it is tuned automatically so that
            the miscoverage budget is met with equality. Increase to favor
            coverage, decrease to favor shorter intervals.
        random_state
            Control the randomness of probabilistic conformal forecasts
            (sample generation) across different runs.
        """
        if not symmetric:
            raise_log(
                ValueError(
                    "`ConformalBellmanModel` only supports `symmetric=True` "
                    "(symmetric non-conformity scores)."
                ),
            )
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=True,
            cal_length=cal_length,
            cal_stride=cal_stride,
            cal_num_samples=cal_num_samples,
            random_state=random_state,
        )
        # target long-run miscoverage from the innermost interval coverage;
        # `interval_range` is sorted from widest to narrowest
        self.alpha0 = 1.0 - self.interval_range[0]
        self.lam = lam
        if lam is not None and lam < 0.0:
            raise_log(ValueError("`lam` must be `>=0` or `None`."))

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calibrate interval bounds by allocating miscoverage across the horizon.

        `residuals` has shape (horizon, n components, n historical forecasts *
        n samples); each horizon step carries the scores for that step only.
        """
        horizon, n_comps, _ = residuals.shape
        n_intervals = len(self.q_interval)
        # solve the control problem per (component, interval): the paper
        # allocates one shared miscoverage budget per coverage level
        widths = np.empty((horizon, n_comps, n_intervals))
        for i, coverage in enumerate(self.interval_range):
            alpha0 = 1.0 - coverage
            for c in range(n_comps):
                widths[:, c, i] = bellman_interval_widths(
                    scores=np.abs(residuals[:, c, :]),
                    alpha0=alpha0,
                    lam=self.lam,
                )
        # `q_interval` is ordered from widest to narrowest; enforce nested
        # intervals so each bound envelops the narrower ones
        widths = np.sort(widths, axis=2)[:, :, ::-1]
        return -widths, widths[:, :, ::-1]
