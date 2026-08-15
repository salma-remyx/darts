"""
Bellman Conformal Calibration
-----------------------------

Conformal interval calibration that plans the per-step miscoverage rate over the forecast
horizon by solving a one-dimensional stochastic control problem with dynamic programming.

This is a clean-room implementation of the algorithm from "Bellman Conformal Inference:
Calibrating Prediction Intervals For Time Series" (Yang, Candès, Lei),
https://arxiv.org/abs/2402.05203.
"""

import numpy as np

from darts.logging import raise_log
from darts.models.forecasting.conformal_models import ConformalNaiveModel

__all__ = [
    "ConformalBellmanModel",
    "bellman_interval_radii",
]


def _validity_weight(
    scores: np.ndarray,
    miscoverage: float,
    step_size: float,
    max_weight: float,
    weight_window: int,
) -> float:
    """Reconstruct the current validity weight from the calibration scores.

    In the paper the weight ``lambda_t`` is carried across forecasts by an online update on the
    realized miscoverage. A conformal model here is a stateless wrapper around `predict()`, so
    there is no carried state; instead the update is replayed over the most recent calibration
    scores. Each score is checked against the conformal radius calibrated only on the scores
    observed before it, and the weight accumulates the excess miscoverage. Under recent
    under-coverage (e.g. after a distribution shift) the reconstructed weight is positive, which
    widens the intervals of the next forecast.

    Parameters
    ----------
    scores
        Non-conformity scores of a single horizon step, shape ``(n_scores,)``, in temporal order.
    miscoverage
        The target miscoverage rate ``alpha_bar``.
    step_size
        The relative step size ``gamma``.
    max_weight
        The upper bound ``lambda_max``.
    weight_window
        The number of most recent scores to replay.

    Returns
    -------
    float
        The reconstructed weight ``lambda_t``, clipped to ``[0, max_weight]``.
    """
    weight = 0.0
    # replay the online update over the most recent scores; each is scored against the
    # conformal radius calibrated only on the scores observed before it
    start = max(2, len(scores) - weight_window)
    radii = np.array([
        np.quantile(scores[:idx], 1.0 - miscoverage, method="higher")
        for idx in range(start, len(scores))
    ])
    errors = (scores[start:] > radii).astype(float)
    # closed form of the online update `lambda <- lambda + gamma * (err - alpha_bar)` replayed
    # over `errors`
    weight = step_size * np.sum(errors - miscoverage)
    return float(np.clip(weight, 0.0, max_weight))


def bellman_interval_radii(
    scores: np.ndarray,
    miscoverage: float,
    step_size: float = 0.05,
    max_weight: float = 50.0,
    weight_window: int = 100,
) -> np.ndarray:
    """Calibrate adaptive interval radii over a multi-step horizon.

    At each horizon step the miscoverage rate is treated as the action of a one-dimensional
    stochastic control problem: the total interval length over the horizon is minimized while
    a hinge penalty keeps the average miscoverage below ``miscoverage``. The optimal policy is
    computed by the Bellman backward recursion, and the planned trajectory is rolled out from
    the initial (error-free) state.

    Parameters
    ----------
    scores
        Non-negative non-conformity scores with shape ``(horizon, n_scores)``, where the second
        axis holds the calibration scores of each horizon step in temporal order.
    miscoverage
        The target miscoverage rate ``alpha_bar`` of the interval, in ``(0, 1)``.
    step_size
        The relative step size ``gamma`` of the online validity weight update.
    max_weight
        The initial upper bound ``lambda_max`` on the validity weight, relative to the level
        implied by the recent miscoverage feedback. The bound doubles until the planned average
        miscoverage meets the target (guarded by a finite limit).
    weight_window
        The number of most recent calibration scores used to reconstruct the validity weight.

    Returns
    -------
    np.ndarray
        The calibrated interval radius of each horizon step, shape ``(horizon,)``.
    """
    if not 0.0 < miscoverage < 1.0:
        raise_log(ValueError("`miscoverage` must be in `(0, 1)`."))
    scores = np.atleast_2d(np.asarray(scores, dtype=float))
    horizon, n_scores = scores.shape
    if n_scores < 1:
        raise_log(ValueError("`scores` must contain at least one calibration example."))

    # candidate actions: the sorted scores themselves. Deploying the k-th smallest score as
    # radius gives a nominal miscoverage of (n - k) / (n + 1), which doubles as the probability
    # that a future score is *not* covered by that candidate
    sorted_scores = np.sort(scores, axis=1)  # (horizon, n_scores)
    lengths = 2.0 * sorted_scores
    error_prob = (n_scores - np.arange(n_scores)) / (n_scores + 1)

    def solve(weight: float) -> np.ndarray:
        """Solve the control problem for a validity `weight` and roll out the optimal policy."""
        # value function over the cumulative miscoverage count within the horizon,
        # `rho in {0..T}`; terminal cost: hinge on the average excess miscoverage
        counts = np.arange(horizon + 1)
        j_next = weight * np.maximum(counts / horizon - miscoverage, 0.0)

        # backward recursion; `policies[step, rho]` is the candidate index of the optimal action.
        # at the last state `rho=T` no further miscoverage can be accumulated, so its marginal
        # cost is zero
        policies = np.empty((horizon, horizon + 1), dtype=int)
        for step in range(horizon - 1, -1, -1):
            # marginal cost of one additional miscoverage at each state
            d_rho = np.diff(j_next, append=j_next[-1])
            objective = lengths[step][None, :] + d_rho[:, None] * error_prob[None, :]
            policies[step] = np.argmin(objective, axis=1)
            j_next = j_next + np.min(objective, axis=1)

        # roll out the planned trajectory from the initial error-free state; the miscoverage
        # count evolves in expectation, so it is tracked as a fractional count
        radii_ = np.empty(horizon)
        rho = 0.0
        for step in range(horizon):
            action = policies[step, min(int(rho), horizon)]
            radii_[step] = sorted_scores[step, action]
            rho += error_prob[action]
        return radii_

    def mean_error(radii_: np.ndarray) -> float:
        """Expected average miscoverage of the horizon under the deployed radii."""
        covered = np.mean(radii_[:, None] >= sorted_scores, axis=1)
        return float(1.0 - np.mean(covered))

    # In the paper the validity weight `lambda_t` is tuned online, forecast after forecast, until
    # the realized miscoverage sits at the target. A conformal model here is a stateless wrapper
    # around `predict()`, so instead of carrying that state the equilibrium weight is recovered
    # directly: a bisection finds the smallest weight at which the planned average miscoverage
    # does not exceed the target. The weight is anchored at a base level derived from the recent
    # miscoverage feedback (via `_validity_weight`, scaled to the scores to keep the trade-off
    # scale-free), so the intervals widen when recent forecasts have been under-covering.
    feedback = _validity_weight(
        scores.mean(axis=0), miscoverage, step_size, max_weight, weight_window
    )

    # `solve` yields wider intervals for larger weights. Grow the weight (geometrically, capped
    # at `max_weight` times the recent feedback level, or a finite guard when the feedback is
    # quiet) until the planned miscoverage meets the target, then bisect for the smallest such
    # weight: the shortest intervals that still achieve the target coverage. If the cap is
    # reached first, deploy it — the safeguard of the paper.
    anchor = max(feedback, 1.0 / max_weight, 1e-12)
    lo, hi = 0.0, anchor * max_weight
    for _ in range(64):
        if mean_error(solve(hi)) <= miscoverage:
            break
        hi *= 2.0
        if hi > 1e12:
            break
    for _ in range(24):
        mid = 0.5 * (lo + hi)
        if not lo < mid < hi:
            break
        if mean_error(solve(mid)) > miscoverage:
            lo = mid
        else:
            hi = mid
    return solve(hi)


class ConformalBellmanModel(ConformalNaiveModel):
    def __init__(
        self,
        model,
        quantiles: list[float],
        cal_length: int | None = None,
        cal_stride: int = 1,
        cal_num_samples: int = 500,
        step_size: float = 0.05,
        max_weight: float = 50.0,
        weight_window: int = 100,
        random_state: int | None = None,
    ):
        """Bellman Conformal Prediction Model.

        A probabilistic model that adds calibrated intervals around the median forecast from a
        pre-trained global forecasting model. Unlike :class:`~darts.models.forecasting.
        conformal_models.ConformalNaiveModel`, which calibrates each step of the horizon
        independently, this model plans the miscoverage rate of all steps jointly: at each
        forecast it solves a one-dimensional stochastic control problem over the horizon and
        deploys the interval lengths that minimize the average length while keeping the average
        miscoverage of the horizon at the target level. An adaptive validity weight rises when
        recent intervals have been missing, which keeps the long-run coverage calibrated under
        distribution shift.

        Non-conformity scores: uses metric `ae()` (see absolute error
        :func:`~darts.metrics.metrics.ae`) to compute the non-conformity scores on the
        calibration set. Only symmetric intervals are supported.

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
            [0.1, 0.2, 0.5, 0.8, 0.9] correspond to two intervals with (0.9 - 0.1) = 80%, and
            (0.8 - 0.2) 60% coverage around the median (model forecast).
        cal_length
            The number of past forecast errors / non-conformity scores to use as calibration for
            each conformal forecast (and each step in the horizon). If `None`, considers all
            scores.
        cal_stride
            The stride to apply when computing the historical forecasts and non-conformity
            scores on the calibration set. The actual conformal forecasts can have a different
            stride given with parameter `stride` in downstream tasks (e.g. historical forecasts,
            backtest, ...)
        cal_num_samples
            The number of samples to generate for each calibration forecast (if `model` is a
            probabilistic forecasting model). The non-conformity scores are computed on the
            quantile values of these forecasts (using quantiles `quantiles`). Uses `1` for
            deterministic models. The actual conformal forecasts can have a different number of
            samples given with parameter `num_samples` in downstream tasks (e.g. predict,
            historical forecasts, ...).
        step_size
            The relative step size of the adaptive validity weight update. Larger values react
            faster to recent miscoverage.
        max_weight
            The initial upper bound on the validity weight, relative to the level implied by the
            recent miscoverage feedback. The bound doubles until the planned average
            miscoverage meets the target.
        weight_window
            The number of most recent calibration scores used to reconstruct the validity
            weight.
        random_state
            Control the randomness of probabilistic conformal forecasts (sample generation)
            across different runs.

        Examples
        --------
        >>> from darts.models import ConformalBellmanModel, LinearRegressionModel
        >>> from darts.datasets import AirPassengersDataset
        >>> series = AirPassengersDataset().load()
        >>> model = LinearRegressionModel(lags=4, output_chunk_length=3).fit(series)
        >>> conformal_model = ConformalBellmanModel(model=model, quantiles=[0.1, 0.5, 0.9])
        >>> pred = conformal_model.predict(n=3, num_samples=1)  # doctest: +SKIP
        """
        super().__init__(
            model=model,
            quantiles=quantiles,
            symmetric=True,
            cal_length=cal_length,
            cal_stride=cal_stride,
            cal_num_samples=cal_num_samples,
            random_state=random_state,
        )
        if step_size <= 0:
            raise_log(ValueError("`step_size` must be `>0`."))
        if max_weight <= 0:
            raise_log(ValueError("`max_weight` must be `>0`."))
        if weight_window < 1:
            raise_log(ValueError("`weight_window` must be `>=1`."))
        self.step_size = step_size
        self.max_weight = max_weight
        self.weight_window = weight_window

    def _calibrate_interval(
        self, residuals: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        # residuals shape (horizon, n components, n past forecasts), from metric `ae()`
        horizon, n_comps, _ = residuals.shape
        n_intervals = len(self.interval_range)

        # radii shape (horizon, n components, n intervals)
        radii = np.empty((horizon, n_comps, n_intervals))
        for comp in range(n_comps):
            for interval in range(n_intervals):
                radii[:, comp, interval] = bellman_interval_radii(
                    residuals[:, comp, :],
                    miscoverage=1.0 - self.interval_range[interval],
                    step_size=self.step_size,
                    max_weight=self.max_weight,
                    weight_window=self.weight_window,
                )
        return -radii, radii[:, :, ::-1]
