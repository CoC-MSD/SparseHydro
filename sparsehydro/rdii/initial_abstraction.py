"""Physics-based Initial Abstraction (IA) recovery and depletion model.

Tracks available soil/vadose-zone storage capacity over time.  During dry
intervals the capacity recovers toward ``ia_max`` via an exponential approach
whose rate is temperature-dependent (capturing ET and drainage).  During
rainfall the capacity is depleted exponentially per mm of input.  Recovery is
suppressed entirely below ``T_freeze``, producing emergent seasonal RDII
behaviour from a single parameter set.

Key equations
-------------
Recovery (dry interval Δt, temperature T):

    k_rec(T) = k0 + kT * exp(θ * (T - T_ref))   if T >= T_freeze
             = 0                                   if T < T_freeze

    IA_avail(t+Δt) = IA_max - (IA_max - IA_avail(t)) * exp(-k_rec(T) * Δt)

Depletion and rainfall excess (rainfall pulse ΔP mm, uniform within the step):

The wet step integrates the depletion ODE ``d(IA)/dp = -k_dep * IA`` exactly
over the step (p = cumulative rain), so results are invariant to sub-step
refinement — equivalent to uniformly disaggregating the step to an arbitrarily
fine timestep:

    IA(p) = IA_avail(t) * exp(-k_dep * p)

    instantaneous excess rate = max(0, 1 - k_dep * IA(p))

    IA_avail(t+Δt) = IA_avail(t) * exp(-k_dep * ΔP)

Closed form (see :func:`_wet_step_excess`): when ``k_dep * IA_avail <= 1`` the
excess is ``ΔP - IA_avail * (1 - exp(-k_dep * ΔP))``; otherwise no excess is
produced until the within-step rain depth ``p* = ln(k_dep * IA_avail) / k_dep``
has been absorbed.  ``k_dep`` controls how aggressively the remaining capacity
is applied to a given storm (rule of thumb: k_dep ≈ 1 / IA_max; both
k_dep → 0 and k_dep → ∞ reduce total absorption — the interior maximizes it).

Optional degree-day snow model (``snow=True``):

    T <= snow_T :  SWE += ΔP ;  liquid input = 0          (snowfall)
    T >  snow_T :  melt = min(SWE, snow_ddf * (T - snow_T) * Δt_days)
                   SWE -= melt ;  liquid input = ΔP + melt (rain-on-snow adds)

The liquid input replaces raw rainfall in the IA wet/dry logic, so cold-season
precipitation is stored and released during warm spells — producing melt-driven
flow on days with little or no rain.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IModel
from ..parameters import FieldRecord, ScalarParameter


def _wet_step_excess(ia0: float, k_dep: float, P: float) -> float:
    """Exact rainfall excess for one wet step (uniform intra-step rainfall).

    Integrates the IA depletion ODE ``d(ia)/dp = -k_dep * ia`` within the step,
    so the result is invariant to sub-step refinement (it equals the limit of
    uniformly disaggregating the step to an arbitrarily fine timestep).

    - ``k_dep * ia0 <= 1``: the bucket never absorbs the full rain rate, so
      ``excess = P - ia0 * (1 - exp(-k_dep * P))`` (identical to the lumped form).
    - ``k_dep * ia0 > 1``: no excess until the within-step depth
      ``p* = ln(k_dep * ia0) / k_dep`` has fallen, then
      ``excess = (P - p*) - (1/k_dep - ia0 * exp(-k_dep * P))``.

    :param ia0: Available capacity at the start of the step [mm].
    :param k_dep: Depletion rate [1/mm].
    :param P: Rainfall depth in the step [mm].
    :returns: Rainfall excess [mm], in [0, P].
    """
    if P <= 0.0:
        return 0.0
    if k_dep * ia0 <= 1.0:
        return P - ia0 * (1.0 - math.exp(-k_dep * P))
    p_star = math.log(k_dep * ia0) / k_dep
    if p_star >= P:
        return 0.0
    return (P - p_star) - (1.0 / k_dep - ia0 * math.exp(-k_dep * P))


class IAModel(IModel):
    """Stateful initial-abstraction model implementing the :class:`~sparsehydro.interfaces.IModel` lifecycle.

    Constructor arguments seed the :meth:`initialize` parameter registry.
    After :meth:`initialize`, calibratable values live in the scalar-parameter
    registry and are kept in sync with the instance attributes used by the
    physics methods.

    :param ia_max: Maximum abstraction capacity.  Defaults to 0.2 in (imperial)
        or 5.0 mm (metric).  Pass ``None`` to use the unit-appropriate default.
    :param k0: Base recovery rate constant [1/hr].
    :param kT: Temperature-dependent recovery coefficient [1/hr].
    :param theta: Temperature sensitivity exponent [1/°C].
    :param T_ref: Reference temperature [°C].
    :param k_dep: Depletion rate constant.  Defaults to 7.62 /in (imperial)
        or 0.3 /mm (metric).  Pass ``None`` to use the unit-appropriate default.
    :param T_freeze: Temperature below which recovery is suppressed [°C].
    :param units: Unit system — ``"imperial"`` (default, inches) or ``"metric"`` (mm).
    :param snow: Enable the degree-day snow model.  When ``True``, registers
        the calibratable parameters ``snow_T`` (rain/snow partition and melt
        base [°C]) and ``snow_ddf`` (degree-day melt factor [depth/(°C·day)]).
        Precipitation on days with ``T <= snow_T`` accumulates as snow-water
        equivalent and is released as melt during warm spells.
    :param snow_T: Initial rain/snow threshold & melt base [°C].
    :param snow_ddf: Initial degree-day factor.  Defaults to 0.12 in/(°C·day)
        (imperial) or 3.0 mm/(°C·day) (metric).
    """

    model_name = "initial-abstraction"

    def __init__(
        self,
        ia_max: float | None = None,
        k0: float = 0.05,
        kT: float = 0.02,
        theta: float = 0.1,
        T_ref: float = 20.0,
        k_dep: float | None = None,
        T_freeze: float = 0.0,
        units: str = "imperial",
        snow: bool = False,
        snow_T: float = 1.0,
        snow_ddf: float | None = None,
    ) -> None:
        super().__init__()
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self._units = units
        self._snow = bool(snow)
        self.snow_T = float(snow_T)
        if snow_ddf is not None:
            self.snow_ddf = float(snow_ddf)
        else:
            self.snow_ddf = 0.12 if units == "imperial" else 3.0
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"

        if units == "imperial":
            self.ia_max = float(ia_max) if ia_max is not None else 0.2
            self.k_dep = float(k_dep) if k_dep is not None else 7.62
        else:
            self.ia_max = float(ia_max) if ia_max is not None else 5.0
            self.k_dep = float(k_dep) if k_dep is not None else 0.3

        self.k0 = float(k0)
        self.kT = float(kT)
        self.theta = float(theta)
        self.T_ref = float(T_ref)
        self.T_freeze = float(T_freeze)
        self.ia_avail: float = self.ia_max
        self._prepared_df: pd.DataFrame | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register all IA scalar parameters and advance to INITIALIZED."""
        if self._units == "imperial":
            ia_max_lb, ia_max_ub, ia_max_units = 0.004, 2.0, "in"
            ia_kdep_lb, ia_kdep_ub, ia_kdep_units = 0.25, 127.0, "1/in"
            ia_kdep_desc = "Depletion rate per inch of rainfall"
        else:
            ia_max_lb, ia_max_ub, ia_max_units = 0.1, 50.0, "mm"
            ia_kdep_lb, ia_kdep_ub, ia_kdep_units = 0.01, 5.0, "1/mm"
            ia_kdep_desc = "Depletion rate per mm of rainfall"

        self.register_scalar_parameter(ScalarParameter(
            "ia_max", value=self.ia_max, lower_bound=ia_max_lb, upper_bound=ia_max_ub,
            units=ia_max_units, description="Maximum initial abstraction capacity",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k0", value=self.k0, lower_bound=0.001, upper_bound=1.0,
            units="1/hr", description="Base recovery rate (gravity drainage)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_kT", value=self.kT, lower_bound=0.0, upper_bound=0.5,
            units="1/hr", description="Temperature-sensitive recovery coefficient",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_theta", value=self.theta, lower_bound=0.0, upper_bound=0.5,
            units="1/degC", description="Temperature sensitivity exponent",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_ref", value=self.T_ref, lower_bound=0.0, upper_bound=30.0,
            units="degC", description="Reference temperature for ET scaling",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k_dep", value=self.k_dep, lower_bound=ia_kdep_lb, upper_bound=ia_kdep_ub,
            units=ia_kdep_units, description=ia_kdep_desc,
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_freeze", value=self.T_freeze, lower_bound=-5.0, upper_bound=5.0,
            units="degC", description="Recovery suppressed below this temperature",
        ))

        if self._snow:
            if self._units == "imperial":
                ddf_ub, ddf_units = 0.5, "in/degC/day"
            else:
                ddf_ub, ddf_units = 12.0, "mm/degC/day"
            self.register_scalar_parameter(ScalarParameter(
                "snow_T", value=self.snow_T, lower_bound=-2.0, upper_bound=4.0,
                units="degC", description="Rain/snow partition threshold and melt base",
            ))
            self.register_scalar_parameter(ScalarParameter(
                "snow_ddf", value=self.snow_ddf, lower_bound=0.0, upper_bound=ddf_ub,
                units=ddf_units, description="Degree-day snowmelt factor",
            ))

        # --- Output field metadata ---
        depth_units = "in" if self._units == "imperial" else "mm"
        self.register_output_field(FieldRecord(
            name="datetime",
            description="Simulation time step",
        ))
        self.register_output_field(FieldRecord(
            name=self._excess_col,
            units=depth_units,
            description="Rainfall excess depth after initial abstraction",
            calibratable=False,
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and physical constraint T_freeze < T_ref.

        :returns: ``True`` if all parameters are in bounds and constraints hold.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        T_ref = self.get_scalar_parameter("ia_T_ref").value
        if T_freeze >= T_ref:
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame) -> None:
        """Load forcing data and sync parameters from the registry.

        :param data: DataFrame with columns ``datetime``,
            ``rainfall_in`` (imperial) or ``rainfall_mm`` (metric),
            and optionally ``temperature_c``.
        :type data: pandas.DataFrame
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )

        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)

        T_ref = self.get_scalar_parameter("ia_T_ref").value
        if "temperature_c" not in df.columns:
            df["temperature_c"] = T_ref
        else:
            df["temperature_c"] = df["temperature_c"].fillna(T_ref)

        self._prepared_df = df
        self._sync_from_params()
        self.reset()
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Compute rainfall excess for the prepared time series.

        Parameter values changed after :meth:`prepare` are picked up
        automatically (required by :class:`~sparsehydro.calibration.problem.CalibrationProblem`).

        :returns: DataFrame with columns ``datetime`` and ``p_excess_mm``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_from_params()
        df = self._prepared_df

        diffs = df["datetime"].diff().dropna()
        dt_hours = pd.Timedelta(diffs.median()).total_seconds() / 3600.0
        dt_arr = np.full(len(df), dt_hours)

        excess = IAModel.compute_excess_series(
            rainfall_mm=df[self._rainfall_col].to_numpy(dtype=float),
            dt_hours=dt_arr,
            temperature=df["temperature_c"].to_numpy(dtype=float),
            ia_max=self.ia_max,
            k0=self.k0,
            kT=self.kT,
            theta=self.theta,
            T_ref=self.T_ref,
            k_dep=self.k_dep,
            T_freeze=self.T_freeze,
            snow_ddf=self.snow_ddf if self._snow else 0.0,
            snow_T=self.snow_T if self._snow else None,
        )

        result = pd.DataFrame({
            "datetime": df["datetime"].values,
            self._excess_col: excess,
        })
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release stored forcing data and advance to FINALIZED."""
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Physics methods (stateful, single time-step)
    # ------------------------------------------------------------------

    def recovery_rate(self, temperature: float | None = None) -> float:
        """Compute k_rec(T) = k0 + kT * exp(θ*(T - T_ref)), zeroed below T_freeze.

        :param temperature: Air temperature [°C]. ``None`` defaults to ``T_ref``.
        :returns: Recovery rate [1/hr].
        :rtype: float
        """
        T = self.T_ref if temperature is None else float(temperature)
        if T < self.T_freeze:
            return 0.0
        return self.k0 + self.kT * math.exp(self.theta * (T - self.T_ref))

    def step_dry(self, dt_hours: float, temperature: float | None = None) -> float:
        """Advance ``ia_avail`` over a dry interval of ``dt_hours``.

        :param dt_hours: Duration of dry interval [hr].
        :param temperature: Air temperature [°C]. ``None`` uses ``T_ref``.
        :returns: Updated ``ia_avail`` [mm].
        :rtype: float
        """
        k = self.recovery_rate(temperature)
        deficit = self.ia_max - self.ia_avail
        self.ia_avail = self.ia_max - deficit * math.exp(-k * dt_hours)
        self.ia_avail = max(0.0, min(self.ia_max, self.ia_avail))
        return self.ia_avail

    def step_wet(self, delta_precip_mm: float) -> float:
        """Deplete ``ia_avail`` for a rainfall pulse of ``delta_precip_mm`` mm.

        Drains the storage by the consumed volume
        ``IA_consumed = IA_avail * (1 - exp(-k_dep * ΔP))``, i.e.
        ``IA_avail *= exp(-k_dep * ΔP)``.

        :param delta_precip_mm: Rainfall depth [mm], must be ≥ 0.
        :returns: Updated ``ia_avail`` [mm].
        :rtype: float
        """
        if delta_precip_mm <= 0.0:
            return self.ia_avail
        self.ia_avail = self.ia_avail * math.exp(-self.k_dep * delta_precip_mm)
        self.ia_avail = max(0.0, min(self.ia_max, self.ia_avail))
        return self.ia_avail

    def compute_excess(self, rainfall_mm: float) -> float:
        """Return rainfall excess and deplete ``ia_avail`` accordingly.

        Uses the exact within-step integral of the depletion ODE (see
        :func:`_wet_step_excess`), so the result is invariant to sub-step
        refinement.  The storage drains as ``IA_avail *= exp(-k_dep * P)``
        via :meth:`step_wet`.

        :param rainfall_mm: Rainfall depth for this time step [mm].
        :returns: Rainfall excess [mm].
        :rtype: float
        """
        if rainfall_mm <= 0.0:
            return 0.0
        excess = _wet_step_excess(self.ia_avail, self.k_dep, rainfall_mm)
        self.step_wet(rainfall_mm)
        return excess

    def reset(self) -> None:
        """Reset ``ia_avail`` to ``ia_max`` (fully recovered state)."""
        self.ia_avail = self.ia_max

    # ------------------------------------------------------------------
    # Static vectorized method — hot path for the optimizer
    # ------------------------------------------------------------------

    @staticmethod
    def compute_excess_series(
        rainfall_mm: np.ndarray,
        dt_hours: np.ndarray,
        temperature: np.ndarray | None,
        ia_max: float,
        k0: float,
        kT: float,
        theta: float,
        T_ref: float,
        k_dep: float,
        T_freeze: float,
        snow_ddf: float = 0.0,
        snow_T: float | None = None,
    ) -> np.ndarray:
        """Compute rainfall excess array for a full time series.

        Pure function (no instance state) — safe for parallel optimizer use.
        ``ia_avail`` starts at ``ia_max`` (fully saturated capacity).

        :param rainfall_mm: 1-D array of rainfall depths [mm] per time step.
        :param dt_hours: 1-D array of time-step durations [hr].
        :param temperature: 1-D array of temperatures [°C], or ``None`` to
            use ``T_ref`` for every step.
        :param ia_max: Maximum abstraction capacity [mm].
        :param k0: Base recovery rate [1/hr].
        :param kT: Temperature-dependent recovery coefficient [1/hr].
        :param theta: Temperature sensitivity exponent [1/°C].
        :param T_ref: Reference temperature [°C].
        :param k_dep: Depletion rate [1/mm].
        :param T_freeze: Freeze threshold [°C].
        :param snow_ddf: Degree-day snowmelt factor [mm/(°C·day)].  Ignored
            when ``snow_T`` is ``None``.
        :param snow_T: Rain/snow partition threshold and melt base [°C], or
            ``None`` to disable the snow model (default).
        :returns: 1-D array of rainfall excess values [mm].
        :rtype: numpy.ndarray
        """
        n = len(rainfall_mm)
        excess = np.zeros(n, dtype=float)
        ia_avail = float(ia_max)

        snow_on = snow_T is not None
        swe = 0.0
        use_temp = temperature is not None

        for i in range(n):
            P = float(rainfall_mm[i])
            dt = float(dt_hours[i])
            T = float(temperature[i]) if use_temp else T_ref

            if snow_on:
                if T <= snow_T:
                    # Snowfall: store precip as SWE; no liquid input this step
                    swe += P
                    P = 0.0
                elif swe > 0.0:
                    # Degree-day melt; rain-on-snow adds melt to liquid input
                    melt = min(swe, snow_ddf * (T - snow_T) * dt / 24.0)
                    swe -= melt
                    P += melt

            if P <= 0.0:
                # Dry step: recover
                if T >= T_freeze:
                    k_rec = k0 + kT * math.exp(theta * (T - T_ref))
                else:
                    k_rec = 0.0
                deficit = ia_max - ia_avail
                ia_avail = ia_max - deficit * math.exp(-k_rec * dt)
            else:
                # Wet step: exact within-step integral of the depletion ODE
                # (invariant to sub-step refinement); drain matches the
                # absorbed path exactly.
                excess[i] = _wet_step_excess(ia_avail, k_dep, P)
                ia_avail = ia_avail * math.exp(-k_dep * P)

            ia_avail = max(0.0, min(ia_max, ia_avail))

        return excess

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_from_params(self) -> None:
        """Update instance attributes from the scalar-parameter registry."""
        self.ia_max = self.get_scalar_parameter("ia_max").value
        self.k0 = self.get_scalar_parameter("ia_k0").value
        self.kT = self.get_scalar_parameter("ia_kT").value
        self.theta = self.get_scalar_parameter("ia_theta").value
        self.T_ref = self.get_scalar_parameter("ia_T_ref").value
        self.k_dep = self.get_scalar_parameter("ia_k_dep").value
        self.T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        if self._snow:
            self.snow_T = self.get_scalar_parameter("snow_T").value
            self.snow_ddf = self.get_scalar_parameter("snow_ddf").value
