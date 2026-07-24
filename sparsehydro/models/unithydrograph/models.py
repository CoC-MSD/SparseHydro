"""Native IUnitHydroComponent unit hydrograph models.

Provides three self-contained implementations that follow the SparseHydro
lifecycle without depending on the legacy ``UnitHydrograph`` adapter.

All models return a DataFrame with ``"Q_pred"`` as the predicted-flow column,
so the same ``CalibrationProblem.column_map`` works for both single models and
:class:`~sparsehydro.models.EnsembleModel` composites.

Kernel normalisation: ``sum(get_kernel(dt)) * dt ≈ 1.0``  (units: [1/hr])

Convolution: ``Q = convolve(rain, kernel * A * dt)[:n]``
so ``A`` represents the effective area ratio (stormflow / rain volume).
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd
from scipy.special import gamma as gamma_func

from ...enums import ModelState
from ..base import IUnitHydroComponent
from ...parameters import ScalarParameter

_MAX_STEPS = 864  # 72 h at 5-min intervals


def _infer_dt_hours(data: pd.DataFrame) -> float:
    """Infer the time-step size from the ``datetime`` column.

    :param data: DataFrame containing a ``datetime`` column.
    :type data: pandas.DataFrame
    :returns: Median time-step size [hr]; ``1/12`` (5 min) for < 2 rows.
    :rtype: float
    """
    if len(data) < 2:
        return 1.0 / 12.0
    dt = data["datetime"].diff().median()
    return dt.total_seconds() / 3600.0


def _normalize_kernel(raw: np.ndarray, dt_hours: float) -> np.ndarray:
    """Normalise so ``sum(result) * dt_hours ≈ 1.0``.

    :param raw: Unnormalised kernel ordinates.
    :type raw: numpy.ndarray
    :param dt_hours: Time-step size [hr].
    :type dt_hours: float
    :returns: Normalised kernel [1/hr]; all zeros if the raw sum is non-positive.
    :rtype: numpy.ndarray
    """
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / (total * dt_hours)


def _trim_pad(arr: np.ndarray, n_steps: int | None) -> np.ndarray:
    """Trim or zero-pad *arr* to exactly *n_steps* elements.

    :param arr: Input array.
    :type arr: numpy.ndarray
    :param n_steps: Target length, or ``None`` to leave *arr* unchanged.
    :type n_steps: int | None
    :returns: Array of length *n_steps* (or *arr* unchanged when *n_steps* is ``None``).
    :rtype: numpy.ndarray
    """
    if n_steps is None:
        return arr
    if len(arr) >= n_steps:
        return arr[:n_steps]
    return np.pad(arr, (0, n_steps - len(arr)))


# ---------------------------------------------------------------------------
# Gamma UH
# ---------------------------------------------------------------------------

class GammaUH(IUnitHydroComponent):
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
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 100.0, tt: float = 2.0, tp: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, tt, tp scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars: dict = {}
        self._vectors: dict = {}
        self._constraints: list = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=0.01, upper_bound=50.0, description="Gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the gamma UH kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized gamma UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tt = self.get_scalar_parameter("tt").value
        tp = max(self.get_scalar_parameter("tp").value, 1e-6)
        max_steps = max(int(5 * tp + 1), 20)
        t = np.arange(1, max_steps + 1, dtype=float)
        raw = np.maximum((t / tp) ** tt * np.exp(-t / tp), 0.0)
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


# ---------------------------------------------------------------------------
# Nash cascade UH
# ---------------------------------------------------------------------------

class NashUH(IUnitHydroComponent):
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
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 100.0, n: float = 2.0, k: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._n_init = float(n)
        self._k_init = float(k)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, n, k scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("n", value=self._n_init, lower_bound=0.01, upper_bound=100.0, description="Number of linear reservoirs"))
        self.register_scalar_parameter(ScalarParameter("k", value=self._k_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Storage coefficient in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the Nash cascade kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized Nash cascade UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        n_val = max(self.get_scalar_parameter("n").value, 1e-6)
        k_val = max(self.get_scalar_parameter("k").value, 1e-6)
        max_steps = max(int(5 * n_val * k_val + 1), 20)
        t = np.arange(1, max_steps + 1, dtype=float)
        eps = np.finfo(float).eps
        denom = (k_val**n_val) * gamma_func(n_val)
        raw = np.maximum((t**(n_val - 1)) * np.exp(-t / k_val) / max(denom, eps), 0.0)
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


# ---------------------------------------------------------------------------
# Triangular UH
# ---------------------------------------------------------------------------

class TriangleUH(IUnitHydroComponent):
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
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 100.0, tt: float = 50.0, tp: float = 20.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, tt, tp scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=5.0, upper_bound=1000.0, units="steps", description="Total UH duration in time steps"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=2.0, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and the ``tp < tt`` ordering constraint.

        :returns: ``True`` if all parameters are within bounds and ``tp < tt``.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok and self.get_scalar_parameter("tp").value >= self.get_scalar_parameter("tt").value:
            ok = False
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the triangular UH kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized triangular UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr]; all zeros when ``tp >= tt``.
        :rtype: numpy.ndarray
        """
        tt_val = self.get_scalar_parameter("tt").value
        tp_val = self.get_scalar_parameter("tp").value
        if tp_val >= tt_val:
            return _trim_pad(np.zeros(1), n_steps)
        n = int(np.ceil(tt_val)) + 1
        t = np.arange(n, dtype=float)
        raw = np.zeros(n)
        rising = (t > 0) & (t <= tp_val)
        raw[rising] = t[rising] / tp_val
        falling = (t > tp_val) & (t <= tt_val)
        raw[falling] = 1.0 - (t[falling] - tp_val) / (tt_val - tp_val)
        raw = np.maximum(raw, 0.0)
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


# ---------------------------------------------------------------------------
# Rectangle UH
# ---------------------------------------------------------------------------

class RectangleUH(IUnitHydroComponent):
    """Rectangular (instant-pulse) unit hydrograph.

    Shape: constant response over ``0 < t <= tr``, zero elsewhere.  Represents a
    uniform runoff pulse with no rising/falling limb.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param tr: Pulse duration in time steps.  Bounds [1, 1000].
    :type tr: float
    """

    model_name: ClassVar[str] = "rectangle-uh"
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 100.0, tr: float = 10.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tr_init = float(tr)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, tr scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tr", value=self._tr_init, lower_bound=1.0, upper_bound=1000.0, units="steps", description="Pulse duration in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the rectangular UH kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized rectangular UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tr_val = max(self.get_scalar_parameter("tr").value, 1e-6)
        n = int(np.ceil(tr_val)) + 1
        t = np.arange(n, dtype=float)
        raw = np.zeros(n)
        raw[(t > 0) & (t <= tr_val)] = 1.0
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


# ---------------------------------------------------------------------------
# Decay UH
# ---------------------------------------------------------------------------

class DecayUH(IUnitHydroComponent):
    """Discrete exponential-decay unit hydrograph.

    Shape: ``f(t) ∝ alpha^t`` — a monotonic recession peaking at ``t = 0`` with
    no rising limb.

    :param A: Effective area ratio.  Bounds [0, 1e4].
    :type A: float
    :param alpha: Per-step decay ratio in ``[0, 1)``.  Bounds [0, 0.999999].
    :type alpha: float
    """

    model_name: ClassVar[str] = "decay-uh"
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 100.0, alpha: float = 0.5) -> None:
        super().__init__()
        self._A_init = float(A)
        self._alpha_init = float(alpha)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, alpha scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("alpha", value=self._alpha_init, lower_bound=0.0, upper_bound=1.0 - 1e-6, description="Per-step decay ratio"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the decay UH kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized decay UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        alpha = min(max(self.get_scalar_parameter("alpha").value, 0.0), 1.0 - 1e-9)
        if alpha <= 0.0:
            raw = np.array([1.0])
            return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)
        max_steps = min(_MAX_STEPS, max(int(np.log(1e-3) / np.log(alpha)) + 1, 20))
        t = np.arange(max_steps, dtype=float)
        raw = np.maximum(alpha**t, 0.0)
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


# ---------------------------------------------------------------------------
# Gamma UH with time delay
# ---------------------------------------------------------------------------

class GammaDelayUH(IUnitHydroComponent):
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
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(self, A: float = 1.0, tt: float = 2.0, tp: float = 5.0, td: float = 5.0) -> None:
        super().__init__()
        self._A_init = float(A)
        self._tt_init = float(tt)
        self._tp_init = float(tp)
        self._td_init = float(td)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the A, tt, tp, td scalar parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=0.01, upper_bound=50.0, description="Gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self.register_scalar_parameter(ScalarParameter("td", value=self._td_init, lower_bound=0.0, upper_bound=200.0, units="steps", description="Response delay in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Cache forcing data and coerce the ``datetime`` column.

        :param data: DataFrame with ``datetime`` and ``rain`` columns.
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        """
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the delayed gamma UH kernel and return predicted flow.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._data is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self.get_scalar_parameter("A").value
        dt = _infer_dt_hours(self._data)
        rain = self._data["rain"].values.astype(float)
        kernel = self.get_kernel(dt_hours=dt)
        Q = np.convolve(rain, kernel * A * dt, mode="full")[: len(rain)]
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._data["datetime"].values, "Q_pred": Q})

    def finalize(self) -> None:
        """Release cached forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
        """Return the normalized delayed-gamma UH ordinate array.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        tt = self.get_scalar_parameter("tt").value
        tp = max(self.get_scalar_parameter("tp").value, 1e-6)
        td = max(self.get_scalar_parameter("td").value, 0.0)
        max_steps = min(_MAX_STEPS, max(int(5 * tp + td + 1), 20))
        t = np.arange(1, max_steps + 1, dtype=float)
        ts = t - td
        raw = np.where(ts > 0.0, (ts / tp) ** tt * np.exp(-ts / tp), 0.0)
        raw = np.maximum(raw, 0.0)
        return _trim_pad(_normalize_kernel(raw, dt_hours), n_steps)


__all__ = ["GammaUH", "NashUH", "TriangleUH", "RectangleUH", "DecayUH", "GammaDelayUH"]
