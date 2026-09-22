"""Native IUnitHydroComponent unit hydrograph models.

Provides six self-contained implementations that follow the SparseHydro
lifecycle without depending on the legacy ``UnitHydrograph`` adapter.

All models return a DataFrame with ``"Q_pred"`` as the predicted-flow column,
so the same ``CalibrationProblem.column_map`` works for both single models and
:class:`~sparsehydro.models.EnsembleModel` composites.

Kernel normalisation: ``sum(get_kernel(dt)) * dt ≈ 1.0``  (units: [1/hr])

Convolution: ``Q = convolve(rain, kernel * A * dt)[:n]``
so ``A`` represents the effective area ratio (stormflow / rain volume).

The shared prepare/predict/finalize lifecycle lives in
:class:`~sparsehydro.models.unithydrograph.base.UnitHydrographBase`; each class
below contributes only its parameter registration and kernel shape.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
from scipy.special import gamma as gamma_func

from ...enums import ModelState
from ...parameters import ScalarParameter
from .base import UnitHydrographBase
from .kernels import (
    MAX_STEPS as _MAX_STEPS,
    finalize_kernel,
    kernel_errstate,
    infer_dt_hours as _infer_dt_hours,
    normalize_kernel as _normalize_kernel,
    trim_pad as _trim_pad,
)

__all__ = [
    "GammaUH",
    "NashUH",
    "TriangleUH",
    "RectangleUH",
    "DecayUH",
    "GammaDelayUH",
    # Re-exported for backwards compatibility with the pre-split private names.
    "_MAX_STEPS",
    "_infer_dt_hours",
    "_normalize_kernel",
    "_trim_pad",
]


# ---------------------------------------------------------------------------
# Gamma UH
# ---------------------------------------------------------------------------

class GammaUH(UnitHydrographBase):
    """Gamma-function unit hydrograph.

    Shape: ``f(t) ∝ (t/tp)^tt * exp(-t/tp)``

    :param A: Effective area ratio (stormflow / rain volume).  Bounds [0, 1e4].
    :type A: float
    :param tt: Shape parameter (dimensionless).  Bounds [0.01, 50].
    :type tt: float
    :param tp: Scale / time-to-peak in time steps.  Bounds [0.01, 500].
    :type tp: float
    """

    model_name: ClassVar[str] = "gamma-uh"

    def __init__(self, A: float = 100.0, tt: float = 2.0, tp: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)

    def initialize(self) -> None:
        """Register the A, tt, tp scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=0.01, upper_bound=50.0, description="Gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized gamma UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tt = self._p("tt")
        tp = max(self._p("tp"), 1e-6)
        max_steps = min(_MAX_STEPS, max(int(5 * tp + 1), 20))
        t = np.arange(1, max_steps + 1, dtype=float)
        with kernel_errstate():
            raw = np.maximum((t / tp) ** tt * np.exp(-t / tp), 0.0)
        return finalize_kernel(raw, dt_hours, n_steps)


# ---------------------------------------------------------------------------
# Nash cascade UH
# ---------------------------------------------------------------------------

class NashUH(UnitHydrographBase):
    """Nash cascade (linear-reservoir) unit hydrograph.

    Shape: ``f(t) ∝ t^(n-1) * exp(-t/k)``

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param n: Number of linear reservoirs.  Bounds [0.01, 100].
    :type n: float
    :param k: Storage coefficient in time steps.  Bounds [0.01, 500].
    :type k: float
    """

    model_name: ClassVar[str] = "nash-uh"

    def __init__(self, A: float = 100.0, n: float = 2.0, k: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._n_init = float(n)
        self._k_init = float(k)

    def initialize(self) -> None:
        """Register the A, n, k scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("n", value=self._n_init, lower_bound=0.01, upper_bound=100.0, description="Number of linear reservoirs"))
        self.register_scalar_parameter(ScalarParameter("k", value=self._k_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Storage coefficient in time steps"))
        self._state = ModelState.INITIALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized Nash cascade UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        n_val = max(self._p("n"), 1e-6)
        k_val = max(self._p("k"), 1e-6)
        max_steps = min(_MAX_STEPS, max(int(5 * n_val * k_val + 1), 20))
        t = np.arange(1, max_steps + 1, dtype=float)
        eps = np.finfo(float).eps
        with kernel_errstate():
            denom = (k_val**n_val) * gamma_func(n_val)
            # denom overflows to inf for large n and k (500**100 * gamma(100)),
            # which would zero every ordinate.  It is a constant scale factor
            # that finalize_kernel's normalisation divides out regardless, so
            # dropping it when it is unusable changes nothing but the overflow.
            if not np.isfinite(denom) or denom <= 0.0:
                denom = 1.0
            raw = np.maximum((t**(n_val - 1)) * np.exp(-t / k_val) / max(denom, eps), 0.0)
        return finalize_kernel(raw, dt_hours, n_steps)


# ---------------------------------------------------------------------------
# Triangular UH
# ---------------------------------------------------------------------------

class TriangleUH(UnitHydrographBase):
    """Triangular unit hydrograph.

    Rising limb 0 → peak at ``tp``; falling limb peak → 0 at ``tt``.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param tt: Total UH duration in time steps.  Bounds [5, 1000].
    :type tt: float
    :param tp: Time to peak in time steps.  Bounds [2, 500].  Must be < tt.
    :type tp: float
    """

    model_name: ClassVar[str] = "triangle-uh"

    def __init__(self, A: float = 100.0, tt: float = 50.0, tp: float = 20.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)

    def initialize(self) -> None:
        """Register the A, tt, tp scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=5.0, upper_bound=1000.0, units="steps", description="Total UH duration in time steps"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=2.0, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def _extra_valid(self) -> bool:
        """Return whether the ``tp < tt`` ordering constraint holds.

        :returns: ``True`` when time-to-peak precedes total duration.
        :rtype: bool
        """
        return self._p("tp") < self._p("tt")

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized triangular UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr]; all zeros when ``tp >= tt``.
        :rtype: numpy.ndarray
        """
        tt_val = self._p("tt")
        tp_val = self._p("tp")
        if tp_val >= tt_val:
            return finalize_kernel(np.zeros(1), dt_hours, n_steps)
        n = min(_MAX_STEPS, int(np.ceil(tt_val)) + 1)
        t = np.arange(n, dtype=float)
        raw = np.zeros(n)
        rising = (t > 0) & (t <= tp_val)
        raw[rising] = t[rising] / tp_val
        falling = (t > tp_val) & (t <= tt_val)
        raw[falling] = 1.0 - (t[falling] - tp_val) / (tt_val - tp_val)
        raw = np.maximum(raw, 0.0)
        return finalize_kernel(raw, dt_hours, n_steps)


# ---------------------------------------------------------------------------
# Rectangle UH
# ---------------------------------------------------------------------------

class RectangleUH(UnitHydrographBase):
    """Rectangular (instant-pulse) unit hydrograph.

    Shape: constant response over ``0 < t <= tr``, zero elsewhere.  Represents a
    uniform runoff pulse with no rising/falling limb.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param tr: Pulse duration in time steps.  Bounds [1, 1000].
    :type tr: float
    """

    model_name: ClassVar[str] = "rectangle-uh"

    def __init__(self, A: float = 100.0, tr: float = 10.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tr_init = float(tr)

    def initialize(self) -> None:
        """Register the A, tr scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tr", value=self._tr_init, lower_bound=1.0, upper_bound=1000.0, units="steps", description="Pulse duration in time steps"))
        self._state = ModelState.INITIALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized rectangular UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tr_val = max(self._p("tr"), 1e-6)
        n = min(_MAX_STEPS, int(np.ceil(tr_val)) + 1)
        t = np.arange(n, dtype=float)
        raw = np.zeros(n)
        raw[(t > 0) & (t <= tr_val)] = 1.0
        return finalize_kernel(raw, dt_hours, n_steps)


# ---------------------------------------------------------------------------
# Decay UH
# ---------------------------------------------------------------------------

class DecayUH(UnitHydrographBase):
    """Discrete exponential-decay unit hydrograph.

    Shape: ``f(t) ∝ alpha^t`` — a monotonic recession peaking at ``t = 0`` with
    no rising limb.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param alpha: Per-step decay ratio in ``[0, 1)``.  Bounds [0, 0.999999].
    :type alpha: float
    """

    model_name: ClassVar[str] = "decay-uh"

    def __init__(self, A: float = 100.0, alpha: float = 0.5) -> None:
        super().__init__()
        self._A_init = float(A)
        self._alpha_init = float(alpha)

    def initialize(self) -> None:
        """Register the A, alpha scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("alpha", value=self._alpha_init, lower_bound=0.0, upper_bound=1.0 - 1e-6, description="Per-step decay ratio"))
        self._state = ModelState.INITIALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized decay UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        alpha = min(max(self._p("alpha"), 0.0), 1.0 - 1e-9)
        if alpha <= 0.0:
            raw = np.array([1.0])
            return finalize_kernel(raw, dt_hours, n_steps)
        max_steps = min(_MAX_STEPS, max(int(np.log(1e-3) / np.log(alpha)) + 1, 20))
        t = np.arange(max_steps, dtype=float)
        with kernel_errstate():
            raw = np.maximum(alpha**t, 0.0)
        return finalize_kernel(raw, dt_hours, n_steps)


# ---------------------------------------------------------------------------
# Gamma UH with time delay
# ---------------------------------------------------------------------------

class GammaDelayUH(UnitHydrographBase):
    """Gamma-function unit hydrograph with a pure time delay.

    Shape: ``f(t) ∝ ((t-td)/tp)^tt * exp(-(t-td)/tp)`` for ``t > td``, else 0.
    Captures a routing lag before the gamma response begins.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param tt: Shape parameter (dimensionless).  Bounds [0.01, 50].
    :type tt: float
    :param tp: Scale / time-to-peak in time steps.  Bounds [0.01, 500].
    :type tp: float
    :param td: Delay before response onset, in time steps.  Bounds [0, 200].
    :type td: float
    """

    model_name: ClassVar[str] = "gamma-delay-uh"

    def __init__(self, A: float = 1.0, tt: float = 2.0, tp: float = 5.0, td: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)
        self._td_init = float(td)

    def initialize(self) -> None:
        """Register the A, tt, tp, td scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=0.01, upper_bound=50.0, description="Gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self.register_scalar_parameter(ScalarParameter("td", value=self._td_init, lower_bound=0.0, upper_bound=200.0, units="steps", description="Response delay in time steps"))
        self._state = ModelState.INITIALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized delayed-gamma UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tt = self._p("tt")
        tp = max(self._p("tp"), 1e-6)
        td = max(self._p("td"), 0.0)
        max_steps = min(_MAX_STEPS, max(int(5 * tp + td + 1), 20))
        t = np.arange(1, max_steps + 1, dtype=float)
        ts = t - td
        with kernel_errstate():
            raw = np.where(ts > 0.0, (ts / tp) ** tt * np.exp(-ts / tp), 0.0)
            raw = np.maximum(raw, 0.0)
        return finalize_kernel(raw, dt_hours, n_steps)
