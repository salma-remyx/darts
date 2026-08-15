"""
Dynamic-Programming Conformal Model
-----------------------------------

Conformal prediction model with length-optimized, multi-step-aware interval calibration.
"""

import math

import numpy as np

from darts.logging import raise_log
from darts.models.forecasting.conformal_models import (
    ConformalNaiveModel,
)
from darts.utils.utils import random_method


def solve_bellman_interval_policy(
    scores: np.ndarray,
    alpha_bar: float,
    lam: float,
    lam_max: float,
) -> list[int]:
    """Solve the per-horizon-step stochastic control problem for one target coverage.

    Adapted from the dynamic programming algorithm of "Bellman Conformal Inference:
    Calibrating Prediction Intervals For Time Series" (Yang, Candès, Lei; arXiv:2402.05203).
    At each step in the horizon, a miscoverage level is chosen from the order statistics of the
    calibration scores (the candidate actions), trading interval length against an accumulated
    miscoverage budget:

    - state: the (running) number of miscoverages accumulated over the horizon steps so far,
    - cost: interval half-width (efficiency) plus ``lam`` times the terminal violation of the
      target miscoverage ``alpha_bar`` (validity),
    - policy: backward Bellman recursion over the state, followed by a forward rollout from
      zero accumulated miscoverages.

    The paper's rollout simulates future PIT values from an estimated CDF; here the rollout is
    certainty-equivalent (expected miscoverage of each candidate action), which keeps the policy
    deterministic. With ``lam >= lam_max``, the safeguard action is returned (the widest interval).

    Parameters
    ----------
    scores
        Sorted (ascending) absolute calibration scores per horizon step, with shape
        ``(horizon, n calibration scores)``. The ``j``-th smallest score at a step is the
        half-width of the interval that covers ``j / (n + 1)`` of the calibration scores.
    alpha_bar
        The target miscoverage level (``1 - desired coverage``).
    lam
        Non-negative weight of the coverage-violation cost (from the online update).
    lam_max
        The weight above which the safeguard (widest) interval is used.

    Returns
    -------
    list[int]
        The chosen rank (1-based, into the sorted scores) for each horizon step.
    """
    horizon = len(scores)
    ranks = []
    if lam >= lam_max:
        return [len(scores[h]) for h in range(horizon)]

    # expected miscoverage when emitting the j-th smallest score as half-width
    exp_err = [
        (len(scores[h]) + 1 - np.arange(1, len(scores[h]) + 1)) / (len(scores[h]) + 1)
        for h in range(horizon)
    ]

    # `cost_curves[h]` is the optimal cost-to-go from stage `h` onwards, over the states
    # `rho` (accumulated miscoverages) reachable at stage `h`: `{0, ..., h}`
    cost_curves = [None] * (horizon + 1)
    # terminal cost: `lam` times the violation of the target miscoverage budget
    cost_curves[horizon] = lam * np.maximum(
        np.arange(horizon + 1) / horizon - alpha_bar, 0.0
    )

    # backward recursion: J_h(rho) = min_j [ length(j) + D(rho) * E(err | j) ] + J_{h+1}(rho)
    for h in range(horizon - 1, -1, -1):
        next_cost = cost_curves[h + 1]
        # marginal cost of one additional miscoverage; non-negative since J is non-decreasing
        d_rho = np.maximum(next_cost[1:] - next_cost[:-1], 0.0)
        candidates = scores[h][:, None] + d_rho[None, :] * exp_err[h][:, None]
        cost_curves[h] = next_cost[:-1] + candidates.min(axis=0)

    # forward rollout from zero accumulated miscoverages (certainty-equivalent)
    rho = 0.0
    for h in range(horizon):
        next_cost = cost_curves[h + 1]
        x = np.arange(len(next_cost))
        # marginal cost of one additional miscoverage at the current expected count `rho`
        d_rho = np.interp(rho + 1.0, x, next_cost) - np.interp(rho, x, next_cost)
        j = int(np.argmin(scores[h] + max(d_rho, 0.0) * exp_err[h]))
        ranks.append(j + 1)
        rho += exp_err[h][j]
    return ranks


class ConformalDPModel(ConformalNaiveModel):
    @random_method
    def __init__(
        self,
        model,
        quantiles: list[float],
        cal_length: int | None = None,
        cal_stride: int = 1,
        cal_num_samples: int = 500,
        lam_max: float = 50.0,
        lam_init: float | None = None,
        step_size: float = 0.05,
        random_state: int | None = None,
    ):
        """Dynamic-Programming (length-optimized) Conformal Prediction Model.

        A probabilistic model that adds calibrated intervals around the median forecast from a
        pre-trained global forecasting model, where the miscoverage level of each step in the
        horizon is optimized for interval length by a one-dimensional dynamic programming
        recursion, and kept calibrated over time by an online update.

        Adapted from "Bellman Conformal Inference: Calibrating Prediction Intervals For Time
        Series" (Yang, Candès, Lei; `arXiv:2402.05203 <https://arxiv.org/abs/2402.05203>`__).
        Unlike :class:`~darts.models.forecasting.conformal_models.ConformalNaiveModel`, which uses
        the same quantile of the calibration errors for every step in the horizon, this model
        treats the per-step miscoverage levels as actions of a stochastic control problem: later
        steps in the horizon (typically harder to forecast) receive wider intervals, and steps
        with small calibration errors are "spent" on narrower intervals, such that the average
        interval length is minimized while the coverage target is met. An online update of the
        validity weight adapts the intervals to distribution shifts and poor calibration.

        Substitutions with respect to the paper: the candidate actions are order statistics of
        the absolute calibration errors (instead of an estimated CDF of past PIT values), the
        forward policy rollout is certainty-equivalent (instead of simulated), and the online
        update measures the miscoverage of the emitted interval on the calibration set (instead
        of an observed PIT value). Only symmetric non-conformity scores are supported.

        Non-conformity scores: uses metric `ae()` (see absolute error
        :func:`~darts.metrics.metrics.ae`) to compute the scores on the calibration set.

        Since it is a probabilistic model, you can generate forecasts in two ways (when calling
        `predict()`, `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters
          `predict_likelihood_parameters=True`, and `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Parameters
        ----------
        model
            A pre-trained global forecasting model. See the list of models
            `here <https://unit8co.github.io/darts/#forecasting-models>`__.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.2, 0.5, 0.8 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and
            (0.8 - 0.2) 60% coverage around the median (model forecast). The dynamic program is
            solved independently for each interval (target coverage).
        cal_length
            The number of past forecast errors / non-conformity scores to use as calibration for
            each conformal forecast (and each step in the horizon). If `None`, considers all
            scores. The candidate interval widths are the order statistics of these scores.
        cal_stride
            The stride to apply when computing the historical forecasts and non-conformity scores
            on the calibration set.
        cal_num_samples
            The number of samples to generate for each calibration forecast (if `model` is a
            probabilistic forecasting model).
        lam_max
            The validity weight above which the safeguard interval is used (the widest interval
            that covers all calibration scores). Together with `step_size`, controls how strongly
            the model reacts to under-coverage.
        lam_init
            The initial validity weight (in units of the target series). If `None`, it is set
            automatically from the calibration scores at the first forecast. Must be in
            `[0, lam_max]`.
        step_size
            The relative step size of the online validity-weight update, in `(0, 1)`. Larger
            values adapt the intervals faster (at the cost of stability).
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation)
            across different runs.

        Examples
        --------
        >>> from darts.models import ConformalDPModel, LinearRegressionModel
        >>> model = ConformalDPModel(  # doctest: +SKIP
        ...     model=LinearRegressionModel(lags=4).fit(series),
        ...     quantiles=[0.1, 0.5, 0.9],
        ... )
        >>> pred = model.predict(n=3, series=series)  # doctest: +SKIP
        """
        if lam_max <= 0:
            raise_log(ValueError("`lam_max` must be larger than `0`."))
        if lam_init is not None and not 0.0 <= lam_init <= lam_max:
            raise_log(ValueError("`lam_init` must be in `[0, lam_max]` or `None`."))
        if not 0.0 < step_size < 1.0:
            raise_log(ValueError("`step_size` must be in `(0, 1)`."))
        if not quantiles or len(quantiles) < 3:
            raise_log(
                ValueError("`quantiles` must contain at least one interval and the median.")
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

        self.lam_max = lam_max
        self.lam_init = lam_init
        self.step_size = step_size
        # validity weight per component, updated online after every forecast
        self._lam: np.ndarray | None = None

    def _standard_widths(self, scores: np.ndarray) -> np.ndarray:
        """Half-widths from the standard conformal order statistics, for all intervals.

        Parameters
        ----------
        scores
            Sorted (ascending) absolute calibration scores with shape
            ``(horizon, n components, n calibration scores)``.
        """
        n_scores = scores.shape[2]
        widths = np.empty(
            (scores.shape[0], scores.shape[1], len(self.interval_range))
        )
        for i, cov in enumerate(self.interval_range):
            rank = min(max(math.ceil(cov * (n_scores + 1)), 1), n_scores)
            widths[:, :, i] = scores[:, :, rank - 1]
        return widths

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Computes the lower and upper calibrated forecast intervals based on residuals.

        For each component and interval, the half-width of each step in the horizon is the order
        statistic chosen by the dynamic programming policy, and the validity weight is updated
        online from the miscoverage of the emitted interval on the calibration set.

        Parameters
        ----------
        residuals
            The residuals are expected to have shape
            ``(horizon, n components, n historical forecasts * n samples)``.
        """
        # sorted absolute scores (from metric `ae()`): candidate interval half-widths
        scores = np.sort(np.abs(residuals), axis=2)
        horizon, n_comps, n_scores = scores.shape

        if self._lam is None:
            # data-scaled initial validity weight: mean standard-conformal half-width of the
            # widest interval over the horizon
            base = self._standard_widths(scores)
            self._lam = (
                np.ones(n_comps) * self.lam_init
                if self.lam_init is not None
                else base[:, :, 0].mean(axis=0)
            )

        # (horizon, n components, n intervals) half-widths from the DP policy
        widths = np.empty((horizon, n_comps, len(self.interval_range)))
        for comp in range(n_comps):
            for i, coverage in enumerate(self.interval_range):
                ranks = solve_bellman_interval_policy(
                    scores=scores[:, comp, :],
                    alpha_bar=1.0 - coverage,
                    lam=float(self._lam[comp]),
                    lam_max=self.lam_max,
                )
                for h, rank in enumerate(ranks):
                    widths[h, comp, i] = scores[h, comp, rank - 1]

        # online validity-weight update: miscoverage of the emitted (widest) interval at the
        # first step in the horizon, measured on the calibration set
        alpha_bar = 1.0 - self.interval_range[0]
        err = (np.abs(residuals[0]) > widths[0, :, 0][:, None]).mean(axis=1)
        gamma = self.step_size * max(
            float(np.mean(self._standard_widths(scores)[:, :, 0])), 1e-12
        )
        self._lam = np.clip(self._lam - gamma * (alpha_bar - err), 0.0, self.lam_max)

        return -widths, widths[:, :, ::-1]
