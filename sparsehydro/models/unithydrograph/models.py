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
    if len(data) < 2:
        return 1.0 / 12.0
    dt = data["datetime"].diff().median()
    return dt.total_seconds() / 3600.0


def _normalize_kernel(raw: np.ndarray, dt_hours: float) -> np.ndarray:
    """Normalise so ``sum(result) * dt_hours ≈ 1.0``."""
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / (total * dt_hours)


def _trim_pad(arr: np.ndarray, n_steps: int | None) -> np.ndarray:
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

    Parameters
    ----------
    A : float
        Effective area ratio (stormflow / rain volume).  Bounds [0, 1e4].
    tt : float
        Shape parameter (dimensionless).  Bounds [0.01, 50].
    tp : float
        Scale / time-to-peak in time steps.  Bounds [0.01, 500].
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
        self._scalars: dict = {}
        self._vectors: dict = {}
        self._constraints: list = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=0.01, upper_bound=50.0, description="Gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
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
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
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

    Parameters
    ----------
    A : float
        Effective area ratio.  Bounds [0, 1e4].
    n : float
        Number of linear reservoirs.  Bounds [0.01, 100].
    k : float
        Storage coefficient in time steps.  Bounds [0.01, 500].
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
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("n", value=self._n_init, lower_bound=0.01, upper_bound=100.0, description="Number of linear reservoirs"))
        self.register_scalar_parameter(ScalarParameter("k", value=self._k_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Storage coefficient in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
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
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
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

    Parameters
    ----------
    A : float
        Effective area ratio.  Bounds [0, 1e4].
    tt : float
        Total UH duration in time steps.  Bounds [5, 1000].
    tp : float
        Time to peak in time steps.  Bounds [2, 500].  Must be < tt.
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
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("tt", value=self._tt_init, lower_bound=5.0, upper_bound=1000.0, units="steps", description="Total UH duration in time steps"))
        self.register_scalar_parameter(ScalarParameter("tp", value=self._tp_init, lower_bound=2.0, upper_bound=500.0, units="steps", description="Time to peak in time steps"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok and self.get_scalar_parameter("tp").value >= self.get_scalar_parameter("tt").value:
            ok = False
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        self._data = data.copy()
        self._data["datetime"] = pd.to_datetime(self._data["datetime"])
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
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
        self._data = None
        self._state = ModelState.FINALIZED

    def get_kernel(self, dt_hours: float, n_steps: int | None = None) -> np.ndarray:
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


__all__ = ["GammaUH", "NashUH", "TriangleUH"]
