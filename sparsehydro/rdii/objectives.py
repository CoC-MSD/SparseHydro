"""Objective functions for RDII multi-objective calibration.

All functions operate on 1-D numpy arrays of observed and predicted flow.
They are pure functions with no side effects and no dependencies on model
state, making them safe to call from parallel optimizer workers.
"""

from __future__ import annotations

import numpy as np


def peak_weighted_mse(
    observed: np.ndarray,
    predicted: np.ndarray,
    power: float = 1.0,
) -> float:
    """Peak-weighted mean squared error.

    Weights each squared residual by the observed flow at that step relative
    to the mean observed flow, raised to ``power``, so errors during high-flow
    (peak) events are penalised more heavily than errors during low-flow
    periods.

    .. math::

        w_t = (Q_{obs,t} / \\bar{Q}_{obs})^{p}

        PWMSE = \\frac{\\sum_t w_t (Q_{obs,t} - Q_{pred,t})^2}{\\sum_t w_t}

    ``power=1.0`` (default) gives linear weighting; larger values sharpen the
    focus on peaks (e.g. with ``power=2.0`` a flow at 4x the mean carries 16x
    the weight instead of 4x).  ``power=0.0`` reduces to plain MSE.

    :param observed: 1-D array of observed flow values (non-negative).
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :param power: Exponent applied to the normalised-flow weights (>= 0).
    :type power: float
    :returns: Peak-weighted MSE (lower is better, 0 for perfect prediction).
    :rtype: float
    :raises ValueError: If arrays differ in shape, if ``observed`` is
        all-zero (undefined mean), or if ``power`` is negative.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    if power < 0.0:
        raise ValueError(f"power must be >= 0; got {power!r}")
    mean_obs = float(np.mean(obs))
    if mean_obs == 0.0:
        raise ValueError(
            "observed is all zeros; peak-weighted MSE is undefined."
        )
    weights = (obs / mean_obs) ** power
    w_sum = float(np.sum(weights))
    if w_sum == 0.0:
        return 0.0
    return float(np.sum(weights * (obs - pred) ** 2) / w_sum)


def nash_sutcliffe(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Nash-Sutcliffe Efficiency (NSE).

    .. math::

        NSE = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                        {\\sum_t (Q_{obs,t} - \\bar{Q}_{obs})^2}

    NSE = 1.0 for a perfect predictor, 0.0 for the mean-flow predictor,
    and negative for predictions worse than the mean.

    :param observed: 1-D array of observed flow values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :returns: NSE in (−∞, 1.0] (higher is better).
    :rtype: float
    :raises ValueError: If arrays differ in shape, or if ``SS_tot == 0``
        (constant observed series).
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    if ss_tot == 0.0:
        raise ValueError(
            "SS_tot is zero; observed series is constant — NSE is undefined."
        )
    return 1.0 - ss_res / ss_tot


def pbias(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Absolute percent bias.

    Measures the tendency of predicted values to be larger or smaller than
    observed values, expressed as a percentage of total observed volume.
    Returns the absolute value so lower is always better.

    .. math::

        PBIAS = 100 \\cdot \\left|
            \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})}{\\sum_t Q_{obs,t}}
        \\right|

    :param observed: 1-D array of observed flow values (non-negative).
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :returns: Absolute percent bias in [0, ∞) (lower is better, 0 is perfect).
    :rtype: float
    :raises ValueError: If arrays differ in shape, or ``sum(observed) == 0``.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        raise ValueError("sum(observed) is zero; PBIAS is undefined.")
    return float(abs(100.0 * np.sum(obs - pred) / total_obs))


def log_nse(
    observed: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 0.01,
) -> float:
    """Nash-Sutcliffe Efficiency on log-transformed flows (Log-NSE).

    Applying the log transform de-emphasises large peak errors and gives more
    weight to low-flow reproduction — useful when baseflow separation or dry-
    weather infiltration is important.

    .. math::

        \\text{LogNSE} = 1 - \\frac{
            \\sum_t (\\ln(Q_{obs,t} + \\varepsilon) - \\ln(Q_{pred,t} + \\varepsilon))^2
        }{
            \\sum_t (\\ln(Q_{obs,t} + \\varepsilon) - \\overline{\\ln(Q_{obs} + \\varepsilon)})^2
        }

    :param observed: 1-D array of observed flow values (non-negative).
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :param epsilon: Small offset added before taking the log to avoid
        ``log(0)``.  Default ``0.01``.
    :type epsilon: float
    :returns: Log-NSE in (−∞, 1.0] (higher is better).
    :rtype: float
    :raises ValueError: If arrays differ in shape, ``epsilon <= 0``, or the
        log-transformed observed series is constant.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    if epsilon <= 0.0:
        raise ValueError(f"epsilon must be > 0; got {epsilon!r}")
    log_obs = np.log(obs + epsilon)
    log_pred = np.log(np.clip(pred, 0.0, None) + epsilon)
    ss_res = float(np.sum((log_obs - log_pred) ** 2))
    ss_tot = float(np.sum((log_obs - np.mean(log_obs)) ** 2))
    if ss_tot == 0.0:
        raise ValueError(
            "SS_tot is zero; log-transformed observed series is constant — "
            "Log-NSE is undefined."
        )
    return float(1.0 - ss_res / ss_tot)


def volume_relative_error(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Relative total volume error.

    Measures the fractional difference between predicted and observed total
    runoff volume.  Returns the absolute value so lower is always better.

    .. math::

        VRE = \\left|
            \\frac{\\sum_t Q_{pred,t} - \\sum_t Q_{obs,t}}{\\sum_t Q_{obs,t}}
        \\right|

    :param observed: 1-D array of observed flow values (non-negative).
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :returns: Relative volume error in [0, ∞) (lower is better, 0 is perfect).
    :rtype: float
    :raises ValueError: If arrays differ in shape, or ``sum(observed) == 0``.
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    total_obs = float(np.sum(obs))
    if total_obs == 0.0:
        raise ValueError("sum(observed) is zero; volume relative error is undefined.")
    return float(abs(float(np.sum(pred)) - total_obs) / total_obs)


def concordance_correlation_coefficient(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Lin's Concordance Correlation Coefficient (CCC).

    Measures the agreement between observed and predicted values by combining
    both precision (Pearson correlation) and accuracy (closeness to the
    45-degree line of perfect concordance).

    .. math::

        \\rho_c = \\frac{2\\,\\sigma_{op}}
                        {\\sigma_o^2 + \\sigma_p^2 + (\\mu_o - \\mu_p)^2}

    where :math:`\\sigma_{op}` is the covariance, :math:`\\sigma_o^2` and
    :math:`\\sigma_p^2` are the variances, and :math:`\\mu_o`, :math:`\\mu_p`
    are the means.  CCC = 1 for perfect agreement, −1 for perfect inverse
    agreement, and 0 when there is no linear relationship.

    :param observed: 1-D array of observed flow values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :returns: CCC in [−1, 1] (higher is better, 1 is perfect).
    :rtype: float
    :raises ValueError: If arrays differ in shape, or the denominator is zero
        (both series are constant and equal).
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    mean_obs = float(np.mean(obs))
    mean_pred = float(np.mean(pred))
    var_obs = float(np.var(obs))
    var_pred = float(np.var(pred))
    covariance = float(np.mean((obs - mean_obs) * (pred - mean_pred)))
    denominator = var_obs + var_pred + (mean_obs - mean_pred) ** 2
    if denominator == 0.0:
        raise ValueError(
            "Denominator is zero; both series are constant and equal — "
            "CCC is undefined."
        )
    return float(2.0 * covariance / denominator)


def index_of_agreement(
    observed: np.ndarray,
    predicted: np.ndarray,
) -> float:
    """Willmott's Index of Agreement (d).

    A dimensionless measure of relative model error bounded to [0, 1].
    Unlike NSE it is not sensitive to differences of means and variances, but
    it is also more lenient than NSE for poor models.

    .. math::

        d = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                      {\\sum_t (|Q_{pred,t} - \\bar{Q}_{obs}| +
                                |Q_{obs,t}  - \\bar{Q}_{obs}|)^2}

    :param observed: 1-D array of observed flow values.
    :type observed: numpy.ndarray
    :param predicted: 1-D array of predicted flow values (same length).
    :type predicted: numpy.ndarray
    :returns: Index of Agreement in [0, 1] (higher is better, 1 is perfect).
    :rtype: float
    :raises ValueError: If arrays differ in shape, or the potential error sum
        is zero (all observations and predictions equal the mean).
    """
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    if obs.shape != pred.shape:
        raise ValueError(
            f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
        )
    mean_obs = float(np.mean(obs))
    ss_res = float(np.sum((obs - pred) ** 2))
    ss_pot = float(np.sum((np.abs(pred - mean_obs) + np.abs(obs - mean_obs)) ** 2))
    if ss_pot == 0.0:
        raise ValueError(
            "Potential error (SS_pot) is zero; Index of Agreement is undefined."
        )
    return float(1.0 - ss_res / ss_pot)
