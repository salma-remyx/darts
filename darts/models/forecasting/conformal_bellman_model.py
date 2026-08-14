"""
Bellman Conformal Model
-----------------------

A conformal prediction model that couples the calibration across the forecast horizon by solving
a one-dimensional stochastic control problem with dynamic programming. Adapted from Bellman
Conformal Inference (BCI) by Xu & Xie (2024) [1]_.

.. [1] Xu, C., & Xie, Y. (2024). Bellman Conformal Inference: Calibrating Prediction Intervals
       For Time Series. https://arxiv.org/abs/2402.05203
"""

import numpy as np

from darts import metrics
from darts.metrics.utils import METRIC_TYPE
from darts.models.forecasting.conformal_models import ConformalModel
from darts.models.forecasting.forecasting_model import GlobalForecastingModel


class ConformalBellmanModel(ConformalModel):
    def __init__(
        self,
        model: GlobalForecastingModel,
        quantiles: list[float],
        cal_length: int | None = None,
        cal_stride: int = 1,
        cal_num_samples: int = 500,
        random_state: int | None = None,
    ):
        """Bellman Conformal Prediction Model.

        A probabilistic model that adds calibrated intervals around the median forecast from a pre-trained
        global forecasting model, adapted from Bellman Conformal Inference (BCI, Xu & Xie, 2024;
        https://arxiv.org/abs/2402.05203). It does not have to be trained and can generate calibrated forecasts
        directly using the underlying trained forecasting model.

        Unlike :class:`ConformalNaiveModel`, which calibrates every step in the forecast horizon independently
        (each step gets the same target coverage), this model couples the steps: it solves a one-dimensional
        stochastic control problem (SCP) over the horizon that allocates the allowed miscoverage budget
        non-uniformly across steps to explicitly minimize the average interval length, while keeping the
        *average* coverage over the horizon at the target level. Steps where intervals are cheap (small
        calibration errors) are tightened beyond the target coverage, buying budget that is spent on steps
        where intervals are expensive. As in BCI, each per-step Bellman optimality problem
        ``argmin over radii r of (length(r) + trade_off * miscoverage(r))`` is solved exactly by grid search
        over the observed calibration scores, and the trade-off multiplier is driven to the point where the
        average coverage target is met. Since all intervals of a conformal forecast are issued simultaneously
        (no intermediate realizations can be observed within the horizon), the SCP is solved in its open-loop
        form with a single trade-off multiplier shared across steps; the multiplier is found by bisection
        instead of BCI's online update, which requires sequential feedback.

        Non-conformity scores: uses metric `ae()` (see absolute error :func:`~darts.metrics.metrics.ae`) to
        compute symmetric non-conformity scores on the calibration set.

        Since it is a probabilistic model, you can generate forecasts in two ways (when calling `predict()`,
        `historical_forecasts()`, ...):

        - Predict the calibrated quantile intervals directly: Pass parameters `predict_likelihood_parameters=True`, and
          `num_samples=1` to the forecast method.
        - Predict stochastic samples from the calibrated quantile intervals: Pass parameters
          `predict_likelihood_parameters=False`, and `num_samples>>1` to the forecast method.

        Conformal models can be applied to any of Darts' global forecasting model, as long as the model has been
        fitted before. In general the workflow of the models to produce one calibrated forecast/prediction is as
        follows:

        - Extract a calibration set: The calibration set for each conformal forecast is automatically extracted from
          the most recent past of your input series relative to the forecast start point. The number of calibration
          examples (forecast errors / non-conformity scores) to consider can be defined at model creation with
          parameter `cal_length`. Note that when using `cal_stride>1`, a longer history is required since
          the calibration examples are generated with stridden historical forecasts.
        - Generate historical forecasts on the calibration set (using the forecasting model) with a stride `cal_stride`.
        - Compute the errors/non-conformity scores (as defined above) on these historical forecasts
        - Allocate the miscoverage budget over the horizon by solving the SCP described above, and compute the
          per-step quantile values from the errors / non-conformity scores at the allocated coverage levels.
        - Compute the conformal prediction: Using these quantile values, add calibrated intervals to the forecasting
          model's predictions.

        Some notes:

        - When computing `historical_forecasts()`, `backtest()`, `residuals()`, ... the above is applied for each
          forecast (the forecasting model's historical forecasts are only generated once for efficiency).
        - Coverage is controlled on average over the forecast horizon (not per step as in
          :class:`ConformalNaiveModel`); individual steps can deviate from the target coverage in exchange for
          shorter average interval length.

        Parameters
        ----------
        model
            A pre-trained global forecasting model. See the list of models
            `here <https://unit8co.github.io/darts/#forecasting-models>`__.
        quantiles
            A list of quantiles centered around the median `q=0.5` to use. For example quantiles
            [0.1, 0.2, 0.5, 0.8 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and (0.8 - 0.2) 60% coverage
            around the median (model forecast). The average coverage over the horizon is controlled at each of
            these levels.
        cal_length
            The number of past forecast errors / non-conformity scores to use as calibration for each conformal
            forecast (and each step in the horizon). If `None`, considers all scores.
        cal_stride
            The stride to apply when computing the historical forecasts and non-conformity scores on the calibration
            set. The actual conformal forecasts can have a different stride given with parameter `stride` in downstream
            tasks (e.g. historical forecasts, backtest, ...)
        cal_num_samples
            The number of samples to generate for each calibration forecast (if `model` is a probabilistic forecasting
            model). The non-conformity scores are computed on the quantile values of these forecasts (using quantiles
            `quantiles`). Uses `1` for deterministic models. The actual conformal forecasts can have a different number
            of samples given with parameter `num_samples` in downstream tasks (e.g. predict, historical forecasts, ...).
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation) across different runs.
        """
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=True,
            cal_length=cal_length,
            cal_num_samples=cal_num_samples,
            random_state=random_state,
            cal_stride=cal_stride,
        )

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # residuals shape (horizon, n components, n past forecasts)
        n_horizon, n_comps, _ = residuals.shape
        q_hat = np.empty((n_horizon, n_comps, len(self.interval_range)))
        for idx_interval, cov in enumerate(self.interval_range):
            alpha_target = 1.0 - cov
            for c in range(n_comps):
                grids, exceedance = _score_grids(residuals[:, c, :])
                q_hat[:, c, idx_interval] = _bellman_budget_allocation(
                    grids, exceedance, alpha_target
                )
        return -q_hat, q_hat[:, :, ::-1]

    def _apply_interval(self, pred: np.ndarray, q_hat: tuple[np.ndarray, np.ndarray]):
        # convert stochastic predictions to median
        if pred.shape[2] != 1:
            pred = np.expand_dims(np.quantile(pred, 0.5, axis=2), -1)
        # shape (forecast horizon, n components, n quantiles)
        pred = np.concatenate([pred + q_hat[0], pred, pred + q_hat[1]], axis=2)
        # -> (forecast horizon, n components * n quantiles)
        return pred.reshape(len(pred), -1)

    @property
    def _residuals_metric(self) -> tuple[METRIC_TYPE, dict | None]:
        return metrics.ae, None


def _score_grids(scores: np.ndarray) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per horizon step, the candidate interval radii (unique sorted calibration scores) and their
    empirical exceedance rates (fraction of calibration scores strictly above each radius).

    Parameters
    ----------
    scores
        The calibration scores of one component with shape (horizon, n calibration scores).
    """
    grids, exceedance = [], []
    for step_scores in scores:
        n_scores = len(step_scores)
        grid = np.unique(step_scores)
        n_covered = np.searchsorted(np.sort(step_scores), grid, side="right")
        grids.append(grid)
        exceedance.append((n_scores - n_covered) / n_scores)
    return grids, exceedance


def _bellman_budget_allocation(
    grids: list[np.ndarray],
    exceedance: list[np.ndarray],
    alpha_target: float,
    n_bisect: int = 50,
) -> np.ndarray:
    """Allocates the miscoverage budget `alpha_target` over the forecast horizon by solving BCI's
    one-dimensional stochastic control problem, returning the optimal interval radius per step.

    For a trade-off multiplier `mu`, each step solves its Bellman optimality problem
    ``argmin over candidate radii r of (r + mu * exceedance(r))`` exactly by grid search over the
    observed calibration scores (the optimum always lies among the observed scores). The per-step
    optimal exceedance is non-increasing in `mu`, so the smallest `mu` meeting the average coverage
    target ``mean over steps of exceedance(r) <= alpha_target`` is found by bisection. This yields
    the shortest average interval length among all allocations satisfying the coverage constraint.

    Parameters
    ----------
    grids
        Per horizon step, the candidate interval radii (unique sorted calibration scores).
    exceedance
        Per horizon step, the empirical exceedance rate of each candidate radius.
    alpha_target
        The target average miscoverage rate over the horizon (1 - target average coverage).
    n_bisect
        The number of bisection iterations used to find the trade-off multiplier.
    """
    n_horizon = len(grids)

    def solve(mu: float) -> tuple[np.ndarray, float]:
        radii = np.empty(n_horizon)
        miscov = np.empty(n_horizon)
        for k in range(n_horizon):
            idx = np.argmin(grids[k] + mu * exceedance[k])
            radii[k] = grids[k][idx]
            miscov[k] = exceedance[k][idx]
        return radii, miscov.mean()

    # find a feasible multiplier (average miscoverage at most `alpha_target`); with `mu=0` the
    # tightest radii are chosen (maximal miscoverage), and miscoverage is non-increasing in `mu`
    mu_hi = max(1.0, max(float(grid[-1]) for grid in grids))
    radii, avg_miscov = solve(mu_hi)
    for _ in range(n_bisect):
        if avg_miscov <= alpha_target:
            break
        mu_hi *= 2.0
        radii, avg_miscov = solve(mu_hi)
    if avg_miscov > alpha_target:
        # infeasible target (can only happen with heavily tied scores): return the maximum
        # coverage allocation as best effort
        return radii

    # bisect towards the smallest feasible multiplier, i.e. the shortest feasible intervals
    mu_lo = 0.0
    for _ in range(n_bisect):
        mu_mid = 0.5 * (mu_lo + mu_hi)
        radii_mid, avg_miscov_mid = solve(mu_mid)
        if avg_miscov_mid <= alpha_target:
            mu_hi = mu_mid
            radii = radii_mid
        else:
            mu_lo = mu_mid
    return radii
