"""Objective function abstractions for model calibration.

:class:`IObjective` defines the interface that every calibration objective
must implement.  Concrete subclasses encapsulate both the direction of
optimisation (``minimize``) and the computation of a scalar score from
observed vs predicted arrays.

Built-in objectives
-------------------
+---------------------------+----------+------------------------------------------+
| Class                     | minimize | Description                              |
+===========================+==========+==========================================+
| :class:`MSE`              | True     | Mean squared error                       |
| :class:`RMSE`             | True     | Root mean squared error                  |
| :class:`MAE`              | True     | Mean absolute error                      |
| :class:`PeakWeightedMSE`  | True     | Flow-weighted MSE (peaks penalised)      |
| :class:`PBIAS`            | True     | Absolute percent bias (volume balance)   |
| :class:`VolumeRelativeError` | True  | Relative total runoff volume error       |
| :class:`NashSutcliffe`    | False    | Nash–Sutcliffe efficiency                |
| :class:`LogNSE`           | False    | NSE on log-transformed flows             |
| :class:`KGE`              | False    | Kling–Gupta efficiency                   |
| :class:`IndexOfAgreement` | False    | Willmott's Index of Agreement            |
| :class:`ConcordanceCorrelationCoefficient` | False | Lin's Concordance Correlation Coefficient |
+------------------------------------------+----------+------------------------------------------+

Custom objectives
-----------------
Subclass :class:`IObjective`, set ``name`` and ``minimize``, and implement
:meth:`evaluate`::

    class MyObjective(IObjective):
        name = "my_obj"
        minimize = True

        def evaluate(self, observed, predicted):
            return float(np.sum(np.abs(observed - predicted)))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

import numpy as np


class IObjective(ABC):
    """Abstract calibration objective.

    Subclasses must set the class variables :attr:`name` and :attr:`minimize`
    and implement :meth:`evaluate`.

    :cvar name: Short snake_case identifier (used as column name in result
        DataFrames).
    :cvar minimize: ``True`` if lower scores are better; ``False`` for
        objectives like NSE where higher is better.
    """

    name: ClassVar[str]
    minimize: ClassVar[bool]

    @abstractmethod
    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        """Compute a scalar score from observed and predicted arrays.

        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (same shape).
        :type predicted: numpy.ndarray
        :returns: Scalar objective value.
        :rtype: float
        :raises ValueError: If arrays are incompatible or the score is
            undefined (e.g. constant observed series for NSE).
        """

    def compute(
        self,
        observed: np.ndarray,
        predicted: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> float:
        """Evaluate the objective, optionally restricting to a subset of steps.

        :param observed: 1-D array of observed values.
        :type observed: numpy.ndarray
        :param predicted: 1-D array of predicted values (same shape).
        :type predicted: numpy.ndarray
        :param mask: Boolean array of the same length as *observed*.  Only
            positions where ``mask`` is ``True`` are included in the
            computation.  ``None`` (default) uses all values.
        :type mask: numpy.ndarray or None
        :returns: Scalar objective value computed over the selected subset.
        :rtype: float
        :raises ValueError: If *mask* length does not match *observed*, or if
            all mask values are ``False`` (empty subset).
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if mask is not None:
            m = np.asarray(mask, dtype=bool)
            if m.shape != obs.shape:
                raise ValueError(
                    f"mask shape {m.shape} does not match observed shape {obs.shape}"
                )
            if not m.any():
                raise ValueError("mask selects no elements; cannot compute objective.")
            obs = obs[m]
            pred = pred[m]
        return self.evaluate(obs, pred)


# ---------------------------------------------------------------------------
# Error metrics (minimize = True)
# ---------------------------------------------------------------------------

class MSE(IObjective):
    """Mean squared error.

    :cvar name: ``"mse"``
    :cvar minimize: ``True``
    """

    name = "mse"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.mean((obs - pred) ** 2))


class RMSE(IObjective):
    """Root mean squared error.

    :cvar name: ``"rmse"``
    :cvar minimize: ``True``
    """

    name = "rmse"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.sqrt(np.mean((obs - pred) ** 2)))


class MAE(IObjective):
    """Mean absolute error.

    :cvar name: ``"mae"``
    :cvar minimize: ``True``
    """

    name = "mae"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        return float(np.mean(np.abs(obs - pred)))


class PeakWeightedMSE(IObjective):
    """Peak-weighted mean squared error.

    Weights each squared residual by the observed flow at that step relative
    to the mean observed flow, raised to ``power``, penalising errors during
    high-flow events.

    .. math::

        w_t = (Q_{obs,t} / \\bar{Q}_{obs})^{p}

        PWMSE = \\frac{\\sum_t w_t (Q_{obs,t} - Q_{pred,t})^2}{\\sum_t w_t}

    ``power=1.0`` (default) gives linear weighting; larger values sharpen the
    focus on peaks.  ``power=0.0`` reduces to plain MSE.

    :param power: Exponent applied to the normalised-flow weights (>= 0).
    :cvar name: ``"peak_weighted_mse"``
    :cvar minimize: ``True``
    """

    name = "peak_weighted_mse"
    minimize = True

    def __init__(self, power: float = 1.0) -> None:
        if power < 0.0:
            raise ValueError(f"power must be >= 0; got {power!r}")
        self.power = float(power)

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import peak_weighted_mse
        return peak_weighted_mse(observed, predicted, power=self.power)


# ---------------------------------------------------------------------------
# Efficiency metrics (minimize = False)
# ---------------------------------------------------------------------------

class NashSutcliffe(IObjective):
    """Nash–Sutcliffe Efficiency (NSE).

    .. math::

        NSE = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                        {\\sum_t (Q_{obs,t} - \\bar{Q}_{obs})^2}

    NSE = 1 for a perfect predictor; 0 for the mean-flow predictor;
    negative for worse than the mean.

    :cvar name: ``"nash_sutcliffe"``
    :cvar minimize: ``False``
    """

    name = "nash_sutcliffe"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import nash_sutcliffe
        return nash_sutcliffe(observed, predicted)


class KGE(IObjective):
    """Kling–Gupta Efficiency (KGE).

    Decomposes model error into correlation, bias, and variability components:

    .. math::

        KGE = 1 - \\sqrt{(r-1)^2 + (\\alpha-1)^2 + (\\beta-1)^2}

    where :math:`r` is the Pearson correlation, :math:`\\alpha = \\sigma_p /
    \\sigma_o` is the variability ratio, and :math:`\\beta = \\mu_p / \\mu_o`
    is the bias ratio.  KGE = 1 for a perfect predictor.

    :cvar name: ``"kge"``
    :cvar minimize: ``False``

    :raises ValueError: If ``std(observed) == 0`` or ``mean(observed) == 0``.
    """

    name = "kge"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        if obs.shape != pred.shape:
            raise ValueError(
                f"Shape mismatch: observed {obs.shape} vs predicted {pred.shape}"
            )
        std_obs = float(np.std(obs))
        mean_obs = float(np.mean(obs))
        if std_obs == 0.0:
            raise ValueError("std(observed) == 0; KGE is undefined.")
        if mean_obs == 0.0:
            raise ValueError("mean(observed) == 0; KGE bias ratio is undefined.")
        r = float(np.corrcoef(obs, pred)[0, 1])
        alpha = float(np.std(pred)) / std_obs
        beta = float(np.mean(pred)) / mean_obs
        return float(1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2))


class PBIAS(IObjective):
    """Absolute percent bias.

    Measures the tendency of the model to over- or under-predict total volume,
    expressed as a percentage of observed volume.  The absolute value is
    returned so lower is always better.

    .. math::

        PBIAS = 100 \\cdot \\left|
            \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})}{\\sum_t Q_{obs,t}}
        \\right|

    :cvar name: ``"pbias"``
    :cvar minimize: ``True``
    """

    name = "pbias"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import pbias
        return pbias(observed, predicted)


class VolumeRelativeError(IObjective):
    """Relative total volume error.

    Fractional difference between predicted and observed cumulative runoff
    volume.  The absolute value is returned so lower is always better.

    .. math::

        VRE = \\left|
            \\frac{\\sum_t Q_{pred,t} - \\sum_t Q_{obs,t}}{\\sum_t Q_{obs,t}}
        \\right|

    :cvar name: ``"volume_relative_error"``
    :cvar minimize: ``True``
    """

    name = "volume_relative_error"
    minimize = True

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import volume_relative_error
        return volume_relative_error(observed, predicted)


class LogNSE(IObjective):
    """Nash–Sutcliffe Efficiency on log-transformed flows (Log-NSE).

    The log transform de-emphasises large peak errors and gives more weight to
    low-flow reproduction — complementary to the standard :class:`NashSutcliffe`
    when both peak and low-flow performance matter.

    .. math::

        \\text{LogNSE} = 1 - \\frac{
            \\sum_t (\\ln(Q_{obs,t}+\\varepsilon) - \\ln(Q_{pred,t}+\\varepsilon))^2
        }{
            \\sum_t (\\ln(Q_{obs,t}+\\varepsilon) -
                     \\overline{\\ln(Q_{obs}+\\varepsilon)})^2
        }

    :param epsilon: Offset added before the log transform to avoid
        ``log(0)``.  Default ``0.01``.
    :type epsilon: float
    :cvar name: ``"log_nse"``
    :cvar minimize: ``False``
    """

    name = "log_nse"
    minimize = False

    def __init__(self, epsilon: float = 0.01) -> None:
        self.epsilon = epsilon

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import log_nse
        return log_nse(observed, predicted, epsilon=self.epsilon)


class IndexOfAgreement(IObjective):
    """Willmott's Index of Agreement (d).

    A dimensionless goodness-of-fit measure in [0, 1] that is less sensitive
    to large errors than NSE while still rewarding accurate reproduction of
    timing, phase, and magnitude.

    .. math::

        d = 1 - \\frac{\\sum_t (Q_{obs,t} - Q_{pred,t})^2}
                      {\\sum_t (|Q_{pred,t} - \\bar{Q}_{obs}| +
                                |Q_{obs,t}  - \\bar{Q}_{obs}|)^2}

    :cvar name: ``"index_of_agreement"``
    :cvar minimize: ``False``
    """

    name = "index_of_agreement"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import index_of_agreement
        return index_of_agreement(observed, predicted)


class ConcordanceCorrelationCoefficient(IObjective):
    """Lin's Concordance Correlation Coefficient (CCC).

    Combines precision (Pearson correlation) and accuracy (closeness to the
    perfect-concordance line) into a single coefficient bounded to [−1, 1]:

    .. math::

        \\rho_c = \\frac{2\\,\\sigma_{op}}
                        {\\sigma_o^2 + \\sigma_p^2 + (\\mu_o - \\mu_p)^2}

    CCC = 1 for perfect agreement between observed and predicted.  Unlike
    Pearson correlation alone, CCC penalises systematic bias and scale
    differences, making it sensitive to both random and systematic error.

    :cvar name: ``"ccc"``
    :cvar minimize: ``False``

    :raises ValueError: If both series are constant and equal (denominator zero).
    """

    name = "ccc"
    minimize = False

    def evaluate(self, observed: np.ndarray, predicted: np.ndarray) -> float:
        from sparsehydro.rdii.objectives import concordance_correlation_coefficient
        return concordance_correlation_coefficient(observed, predicted)
