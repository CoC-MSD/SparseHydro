"""Storage-based rainfall-abstraction (tank) models.

Ported from Parsimonious Functions ``tank_models.py``.  A tank accumulates a
wetness proxy from rainfall and drains it over time.  The tank fill fraction
``w = V / V_tank`` linearly interpolates an *effective-area* multiplier between
``ae_max`` (empty tank) and ``ae_min`` (full tank); effective rainfall is
``Ie = R · ae(w)``.

Because ``ae`` is an effective-area gain (typically > 1), the tank carries the
rainfall→runoff volume scaling.  When chaining a tank into
:class:`~sparsehydro.models.AbstractionUHModel`, fix the unit-hydrograph
amplitude ``A = 1`` so the gain is not double-counted.

The tank state is advanced with sub-stepping (``n_substeps`` per time step) for
numerical stability of the drainage integration.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ...parameters import FieldRecord, ScalarParameter
from ...registry import registry
from ..base import IModel


class TankAbstractionModel(IModel):
    """Abstract base for storage-based abstraction models.

    Subclasses register a drainage parameter and implement
    :meth:`_drainage_full` (the full-step drainage volume for a given storage).

    :param V_tank: Tank capacity (wetness-proxy storage).  Bounds [0.1, 1e6].
    :type V_tank: float
    :param ae_min: Effective-area multiplier at a full tank.  Bounds [0, 1e4].
    :type ae_min: float
    :param ae_max: Effective-area multiplier at an empty tank.  Bounds [0, 1e4].
        Must be ``>= ae_min``.
    :type ae_max: float
    :param drain: Drainage parameter value (constant ``Qd`` or coefficient ``k``
        depending on the subclass).
    :type drain: float
    :param units: Unit system — ``"imperial"`` (inches, default) or ``"metric"`` (mm).
    :type units: str
    :param n_substeps: Sub-steps per time step for drainage integration.
    :type n_substeps: int
    """

    #: Name of the subclass drainage parameter (e.g. ``"tank_qd"`` or ``"tank_k"``).
    _drain_param_name: ClassVar[str] = "tank_drain"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1000.0
    _drain_units: ClassVar[str] = ""
    _drain_desc: ClassVar[str] = "Drainage parameter"

    def __init__(
        self,
        V_tank: float = 1000.0,
        ae_min: float = 5.0,
        ae_max: float = 10.0,
        drain: float = 0.1,
        units: str = "imperial",
        n_substeps: int = 5,
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"
        self._V_tank_init = float(V_tank)
        self._ae_min_init = float(ae_min)
        self._ae_max_init = float(ae_max)
        self._drain_init = float(drain)
        self._n_substeps = max(int(n_substeps), 1)
        self._prepared_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register the tank parameters and advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter(
            "tank_V", value=self._V_tank_init, lower_bound=0.1, upper_bound=1e6,
            description="Tank capacity (wetness-proxy storage)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "tank_ae_min", value=self._ae_min_init, lower_bound=0.0, upper_bound=1e4,
            description="Effective-area multiplier at full tank",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "tank_ae_max", value=self._ae_max_init, lower_bound=0.0, upper_bound=1e4,
            description="Effective-area multiplier at empty tank",
        ))
        self.register_scalar_parameter(ScalarParameter(
            self._drain_param_name, value=self._drain_init,
            lower_bound=self._drain_lower, upper_bound=self._drain_upper,
            units=self._drain_units, description=self._drain_desc,
        ))
        depth_units = "in" if self._units == "imperial" else "mm"
        self.register_output_field(FieldRecord(name="datetime", description="Simulation time step"))
        self.register_output_field(FieldRecord(
            name=self._excess_col, units=depth_units,
            description="Effective rainfall depth after tank abstraction",
            calibratable=False,
        ))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and the ``ae_min <= ae_max`` ordering.

        :returns: ``True`` if all parameters are within bounds and ``ae_min <= ae_max``.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok and self.get_scalar_parameter("tank_ae_min").value > self.get_scalar_parameter("tank_ae_max").value:
            ok = False
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data: pd.DataFrame) -> None:
        """Load forcing data.

        :param data: DataFrame with ``datetime`` and the unit-specific rainfall
            column (``rainfall_in`` or ``rainfall_mm``).
        :type data: pandas.DataFrame
        :returns: Nothing.
        :rtype: None
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(f"prepare() data is missing required columns: {sorted(missing)}")
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df["datetime"] = pd.to_datetime(df["datetime"])
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)
        self._prepared_df = df
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Advance the tank over the series and return effective rainfall.

        :returns: DataFrame with columns ``datetime`` and the unit-specific
            excess column (``p_excess_in`` or ``p_excess_mm``).
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")
        df = self._prepared_df
        rain = df[self._rainfall_col].to_numpy(dtype=float)

        V_tank = max(self.get_scalar_parameter("tank_V").value, 1e-9)
        ae_min = self.get_scalar_parameter("tank_ae_min").value
        ae_max = self.get_scalar_parameter("tank_ae_max").value
        drain = self.get_scalar_parameter(self._drain_param_name).value
        n = self._n_substeps

        excess = np.zeros(len(rain))
        V = 0.0
        for i, R in enumerate(rain):
            w = min(max(V / V_tank, 0.0), 1.0)
            ae = ae_max - w * (ae_max - ae_min)
            excess[i] = max(R * ae, 0.0)
            r_sub = R / n
            for _ in range(n):
                V += r_sub
                V -= self._drainage_full(V, drain) / n
                V = min(max(V, 0.0), V_tank)

        result = pd.DataFrame({"datetime": df["datetime"].values, self._excess_col: excess})
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release stored forcing data and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    @abstractmethod
    def _drainage_full(self, V: float, drain: float) -> float:
        """Return the full-step drainage volume for the current storage *V*.

        :param V: Current tank storage.
        :type V: float
        :param drain: Subclass drainage parameter value.
        :type drain: float
        :returns: Volume drained over a full time step.
        :rtype: float
        """
        raise NotImplementedError


@registry.register
class ConstantDrainTank(TankAbstractionModel):
    """``V_C`` tank — constant drainage per time step.

    Drainage law: ``Qd`` (independent of storage).
    """

    model_name: ClassVar[str] = "tank-constant"
    _drain_param_name: ClassVar[str] = "tank_qd"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1000.0
    _drain_desc: ClassVar[str] = "Constant drainage per step"

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return constant drainage ``Qd`` clipped to available storage.

        :param V: Current tank storage.
        :type V: float
        :param drain: Constant drainage ``Qd`` per step.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain, V)


@registry.register
class LinearDrainTank(TankAbstractionModel):
    """``V_lin`` tank — linear-reservoir drainage.

    Drainage law: ``k · V`` (proportional to storage).
    """

    model_name: ClassVar[str] = "tank-linear"
    _drain_param_name: ClassVar[str] = "tank_k"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 1.0
    _drain_desc: ClassVar[str] = "Linear drainage coefficient (k·V)"

    def __init__(self, V_tank: float = 1000.0, ae_min: float = 5.0, ae_max: float = 10.0,
                 drain: float = 0.1, units: str = "imperial", n_substeps: int = 5) -> None:
        super().__init__(V_tank=V_tank, ae_min=ae_min, ae_max=ae_max, drain=drain, units=units, n_substeps=n_substeps)

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return storage-proportional drainage ``k·V``.

        :param V: Current tank storage.
        :type V: float
        :param drain: Linear drainage coefficient ``k``.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain * V, V)


@registry.register
class SqrtDrainTank(TankAbstractionModel):
    """``V_sroot`` tank — square-root drainage.

    Drainage law: ``k · sqrt(V)`` (intermediate between constant and linear).
    """

    model_name: ClassVar[str] = "tank-sqrt"
    _drain_param_name: ClassVar[str] = "tank_k"
    _drain_lower: ClassVar[float] = 1e-6
    _drain_upper: ClassVar[float] = 10.0
    _drain_desc: ClassVar[str] = "Square-root drainage coefficient (k·sqrt(V))"

    def __init__(self, V_tank: float = 1000.0, ae_min: float = 5.0, ae_max: float = 10.0,
                 drain: float = 0.1, units: str = "imperial", n_substeps: int = 5) -> None:
        super().__init__(V_tank=V_tank, ae_min=ae_min, ae_max=ae_max, drain=drain, units=units, n_substeps=n_substeps)

    def _drainage_full(self, V: float, drain: float) -> float:
        """Return square-root drainage ``k·sqrt(V)``.

        :param V: Current tank storage.
        :type V: float
        :param drain: Square-root drainage coefficient ``k``.
        :type drain: float
        :returns: Drainage volume over a full step.
        :rtype: float
        """
        return min(drain * float(np.sqrt(max(V, 0.0))), V)


__all__ = [
    "TankAbstractionModel",
    "ConstantDrainTank",
    "LinearDrainTank",
    "SqrtDrainTank",
]
