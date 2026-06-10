"""RDIIModel — physics-based RDII model combining IA abstraction and N RTK triangles.

The model follows the standard sparsehydro lifecycle
(CREATED → INITIALIZED → VALIDATED → PREPARED → PREDICTED → FINALIZED)
and registers all calibratable parameters as :class:`~sparsehydro.parameters.ScalarParameter`
objects so that the optimizer can discover bounds automatically.

Parameter naming convention
---------------------------
- IA parameters: ``ia_max``, ``ia_k0``, ``ia_kT``, ``ia_theta``,
  ``ia_T_ref``, ``ia_k_dep``, ``ia_T_freeze``
- Area parameter: ``area_acres`` — drainage area [acres] used to convert
  the depth hydrograph (mm) to a flow hydrograph (CFS).
- RTK parameters (N triangles, 1-indexed): ``R_1``, ``T_1``, ``K_1``,
  ``R_2``, ``T_2``, ``K_2``, …

``prepare()`` input DataFrame columns
--------------------------------------
+------------------+----------+---------------------------+
| Column           | Required | Notes                     |
+==================+==========+===========================+
| ``datetime``     | Yes      | Any pandas DatetimeLike   |
| ``rainfall_mm``  | Yes      | Depth per step [mm]       |
| ``flow_cfs``     | No       | Observed flow (optimizer) |
| ``temperature_c``| No       | Falls back to ``ia_T_ref``|
+------------------+----------+---------------------------+
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IModel
from ..parameters import ConstraintRecord, ScalarParameter
from ..registry import registry
from .initial_abstraction import IAModel
from .rtk_triangle import RTKTriangle, triangular_uh

# Default (R, T, K) for each triangle index (cycled if N > 3)
_DEFAULT_RTK = [
    (0.05, 1.0, 1.5),    # fast
    (0.03, 12.0, 2.0),   # medium
    (0.02, 72.0, 3.0),   # slow
]

_FFT_THRESHOLD = 500  # switch to fftconvolve when max(n, m) exceeds this

# depth over 1 acre for 1 hour → CFS
# mm: (43560 ft²/ac) / (304.8 mm/ft) / (3600 s/hr) ≈ 0.03970
# in: (43560 ft²/ac) / (12 in/ft)    / (3600 s/hr) ≈ 1.00833
_MM_AC_PER_HR_TO_CFS = 43560.0 / (304.8 * 3600.0)
_IN_AC_PER_HR_TO_CFS = 43560.0 / (12.0 * 3600.0)


@registry.register
class RDIIModel(IModel):
    """Physics-based RDII model: exponential IA recovery + N RTK triangles.

    :param n_triangles: Number of triangular RTK unit hydrograph components.
        Must be >= 1.  Default is 3.
    :type n_triangles: int

    Usage::

        model = RDIIModel(n_triangles=3)
        model.initialize()
        model.validate()
        model.prepare(df)        # df: datetime, rainfall_mm [, flow_cfs, temperature_c]
        result = model.predict() # DataFrame: datetime, rdii_mm, p_excess_mm
        model.finalize()
    """

    model_name = "rdii"

    def __init__(self, n_triangles: int = 3, units: str = "imperial") -> None:
        super().__init__()
        if n_triangles < 1:
            raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")
        if units not in ("imperial", "metric"):
            raise ValueError(f"units must be 'imperial' or 'metric'; got {units!r}")
        self.n_triangles: int = n_triangles
        self._units = units
        self._rainfall_col = "rainfall_in" if units == "imperial" else "rainfall_mm"
        self._excess_col = "p_excess_in" if units == "imperial" else "p_excess_mm"
        self._depth_col = "rdii_in" if units == "imperial" else "rdii_mm"
        self._depth_to_cfs = _IN_AC_PER_HR_TO_CFS if units == "imperial" else _MM_AC_PER_HR_TO_CFS
        self._triangles: list[RTKTriangle] = []
        self._prepared_df: pd.DataFrame | None = None
        self._p_excess: np.ndarray | None = None
        self._uh_kernels: list[np.ndarray] = []
        self._dt_hours: float = 1.0

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self, n_triangles: int | None = None) -> None:
        """Register all IA and RTK ScalarParameters and advance to INITIALIZED.

        :param n_triangles: Override ``self.n_triangles`` if provided.
        :type n_triangles: int, optional
        """
        if n_triangles is not None:
            if n_triangles < 1:
                raise ValueError(f"n_triangles must be >= 1; got {n_triangles!r}")
            self.n_triangles = n_triangles

        # --- Initial Abstraction parameters ---
        if self._units == "imperial":
            ia_max_v, ia_max_lb, ia_max_ub, ia_max_u = 0.2, 0.004, 2.0, "in"
            ia_kdep_v, ia_kdep_lb, ia_kdep_ub, ia_kdep_u = 7.62, 0.25, 127.0, "1/in"
            ia_kdep_desc = "Depletion rate per inch of rainfall"
        else:
            ia_max_v, ia_max_lb, ia_max_ub, ia_max_u = 5.0, 0.1, 50.0, "mm"
            ia_kdep_v, ia_kdep_lb, ia_kdep_ub, ia_kdep_u = 0.3, 0.01, 5.0, "1/mm"
            ia_kdep_desc = "Depletion rate per mm of rainfall"

        self.register_scalar_parameter(ScalarParameter(
            "ia_max", value=ia_max_v, lower_bound=ia_max_lb, upper_bound=ia_max_ub,
            units=ia_max_u, description="Maximum initial abstraction capacity",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k0", value=0.05, lower_bound=0.001, upper_bound=1.0,
            units="1/hr", description="Base recovery rate (gravity drainage)",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_kT", value=0.02, lower_bound=0.0, upper_bound=0.5,
            units="1/hr", description="Temperature-sensitive recovery coefficient",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_theta", value=0.1, lower_bound=0.0, upper_bound=0.5,
            units="1/degC", description="Temperature sensitivity exponent",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_ref", value=20.0, lower_bound=0.0, upper_bound=30.0,
            units="degC", description="Reference temperature for ET scaling",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_k_dep", value=ia_kdep_v, lower_bound=ia_kdep_lb, upper_bound=ia_kdep_ub,
            units=ia_kdep_u, description=ia_kdep_desc,
        ))
        self.register_scalar_parameter(ScalarParameter(
            "ia_T_freeze", value=0.0, lower_bound=-5.0, upper_bound=5.0,
            units="degC", description="Recovery suppressed below this temperature",
        ))

        # --- Area parameter ---
        self.register_scalar_parameter(ScalarParameter(
            "area_acres", value=100.0, lower_bound=0.01, upper_bound=100_000.0,
            units="acres", description="Drainage area — converts rdii_mm depth to rdii_cfs flow",
        ))

        # --- RTK Triangle parameters ---
        for i in range(1, self.n_triangles + 1):
            R0, T0, K0 = _DEFAULT_RTK[(i - 1) % len(_DEFAULT_RTK)]
            self.register_scalar_parameter(ScalarParameter(
                f"R_{i}", value=R0, lower_bound=0.0, upper_bound=1.0,
                units="-", description=f"Triangle {i}: sewer fraction of P_excess",
            ))
            self.register_scalar_parameter(ScalarParameter(
                f"T_{i}", value=T0, lower_bound=0.1, upper_bound=240.0,
                units="hr", description=f"Triangle {i}: time to peak",
            ))
            self.register_scalar_parameter(ScalarParameter(
                f"K_{i}", value=K0, lower_bound=1.001, upper_bound=10.0,
                units="-", description=f"Triangle {i}: recession-to-peak ratio",
            ))

        # --- Inequality constraint metadata ---
        self.register_inequality_constraint(ConstraintRecord(
            name="sum_R_leq_1",
            description=(
                f"R_1 + … + R_{self.n_triangles} ≤ 1.0  "
                f"(total sewer fraction of P_excess must not exceed 100 %)"
            ),
        ))
        self.register_inequality_constraint(ConstraintRecord(
            name="T_freeze_lt_T_ref",
            description=(
                "ia_T_freeze < ia_T_ref  "
                "(freeze threshold must be below the reference temperature)"
            ),
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameters and physical constraints.

        Enforces ``ia_T_freeze < ia_T_ref``. The ``Σ R_i <= 1.0`` constraint
        is reported via :meth:`inequality_constraints` and enforced by the solver.

        :returns: ``True`` if all constraints are satisfied.
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

    def prepare(self, data: pd.DataFrame, **kwargs: Any) -> None:
        """Load input data, infer dt, fill missing temperature, precompute kernels.

        :param data: DataFrame with columns ``datetime``, ``rainfall_mm``,
            and optionally ``flow_cfs`` / ``temperature_c``.
        :type data: pandas.DataFrame
        :raises ValueError: If required columns are absent or dt cannot be
            inferred.
        """
        required = {"datetime", self._rainfall_col}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )

        df = data.sort_values("datetime").reset_index(drop=True).copy()
        df[self._rainfall_col] = df[self._rainfall_col].fillna(0.0).clip(lower=0.0)

        # Infer dt_hours from median inter-row timedelta
        diffs = df["datetime"].diff().dropna()
        if len(diffs) == 0:
            raise ValueError(
                "Cannot infer dt_hours: DataFrame has fewer than 2 rows."
            )
        median_td = diffs.median()
        self._dt_hours = median_td.total_seconds() / 3600.0
        if self._dt_hours <= 0.0:
            raise ValueError(
                f"Inferred dt_hours = {self._dt_hours:.4f} is not positive. "
                "Ensure the datetime column is sorted and has a uniform step."
            )

        # Fill missing temperature with ia_T_ref
        T_ref = self.get_scalar_parameter("ia_T_ref").value
        if "temperature_c" not in df.columns:
            df["temperature_c"] = T_ref
        else:
            df["temperature_c"] = df["temperature_c"].fillna(T_ref)

        self._prepared_df = df
        self._p_excess = self._compute_ia_excess(df)
        self._sync_triangles()
        self._uh_kernels = [
            triangular_uh(tri, self._dt_hours) for tri in self._triangles
        ]
        self._state = ModelState.PREPARED

    def predict(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        """Convolve P_excess with each RTK triangle and return RDII.

        Always recomputes IA excess and re-syncs triangles before convolution,
        so parameter values changed after ``prepare()`` are picked up
        automatically (required by the general :class:`CalibrationProblem`).

        :returns: DataFrame with columns ``datetime``, ``rdii_cfs``,
            ``rdii_mm``, ``p_excess_mm``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If ``prepare()`` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError(
                "Call prepare(data) before predict()."
            )

        # Recompute IA excess then re-sync triangles (params may have changed)
        self._p_excess = self._compute_ia_excess(self._prepared_df)
        self._sync_triangles()
        n = len(self._p_excess)
        rdii = np.zeros(n, dtype=float)

        for tri, kernel in zip(self._triangles, self._uh_kernels):
            if max(n, len(kernel)) > _FFT_THRESHOLD:
                from scipy.signal import fftconvolve  # type: ignore[import]
                conv = fftconvolve(self._p_excess, kernel, mode="full")[:n]
            else:
                conv = np.convolve(self._p_excess, kernel, mode="full")[:n]
            rdii += tri.R * conv

        rdii = np.clip(rdii, 0.0, None)

        area_acres = self.get_scalar_parameter("area_acres").value
        # rdii is in [depth/hr]; multiply by dt_hours to get depth per timestep,
        # and by depth_to_cfs * area to get instantaneous flow rate [CFS].
        rdii_cfs = rdii * area_acres * self._depth_to_cfs
        rdii_depth = rdii * self._dt_hours

        result = pd.DataFrame({
            "datetime": self._prepared_df["datetime"].values,
            "rdii_cfs": rdii_cfs,
            self._depth_col: rdii_depth,
            self._excess_col: self._p_excess.copy(),
        })
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release stored data and advance to FINALIZED."""
        self._prepared_df = None
        self._p_excess = None
        self._uh_kernels = []
        self._triangles = []
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Constraints
    # ------------------------------------------------------------------

    def inequality_constraints(self) -> list[float]:
        """Inequality constraint residuals for the optimizer.

        Returns ``[Σ R_i - 1.0, ia_T_freeze - ia_T_ref]``.
        A value ≤ 0 means feasible.  Solvers enforce these natively (pymoo)
        or via squared penalty (Platypus / SciPy).

        :returns: List of two residuals.
        :rtype: list[float]
        """
        R_sum = sum(
            self.get_scalar_parameter(f"R_{i}").value
            for i in range(1, self.n_triangles + 1)
        )
        T_freeze = self.get_scalar_parameter("ia_T_freeze").value
        T_ref    = self.get_scalar_parameter("ia_T_ref").value
        return [R_sum - 1.0, T_freeze - T_ref]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_triangles(self) -> None:
        """Rebuild ``_triangles`` list from current ScalarParameter values."""
        self._triangles = [
            RTKTriangle(
                R=self.get_scalar_parameter(f"R_{i}").value,
                T=self.get_scalar_parameter(f"T_{i}").value,
                K=self.get_scalar_parameter(f"K_{i}").value,
            )
            for i in range(1, self.n_triangles + 1)
        ]
        if self._prepared_df is not None:
            self._uh_kernels = [
                triangular_uh(tri, self._dt_hours) for tri in self._triangles
            ]

    def _compute_ia_excess(self, df: pd.DataFrame) -> np.ndarray:
        """Delegate to IAModel.compute_excess_series using current IA parameters."""
        return IAModel.compute_excess_series(
            rainfall_mm=df[self._rainfall_col].to_numpy(dtype=float),
            dt_hours=np.full(len(df), self._dt_hours),
            temperature=df["temperature_c"].to_numpy(dtype=float),
            ia_max=self.get_scalar_parameter("ia_max").value,
            k0=self.get_scalar_parameter("ia_k0").value,
            kT=self.get_scalar_parameter("ia_kT").value,
            theta=self.get_scalar_parameter("ia_theta").value,
            T_ref=self.get_scalar_parameter("ia_T_ref").value,
            k_dep=self.get_scalar_parameter("ia_k_dep").value,
            T_freeze=self.get_scalar_parameter("ia_T_freeze").value,
        )
