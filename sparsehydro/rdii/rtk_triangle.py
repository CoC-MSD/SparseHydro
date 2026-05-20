"""Triangular RTK unit hydrograph for a single RDII component.

Each RTK triangle is defined by three parameters:

- **R** — fraction of rainfall excess that enters the sewer (dimensionless).
- **T** — time to peak [hours].
- **K** — ratio of recession time to time-to-peak (must be > 1).

The unit hydrograph shape is a piecewise-linear triangle:

- Rising limb: linear from 0 to peak over [0, T].
- Falling limb: linear from peak to 0 over [T, T*(1+K)].
- Peak ordinate = 2 / (T * (1+K)), ensuring the area under the curve equals 1.

The ordinate array returned by :func:`triangular_uh` satisfies::

    np.sum(ordinates) * dt_hours ≈ 1.0   (unit hydrograph property)

Convolving this array with a rainfall-excess series [mm] yields RDII depths
[mm] per time step.  The R-scaling is applied during convolution in
:class:`~sparsehydro.rdii.model.RDIIModel`.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IModel
from ..parameters import ScalarParameter

_FFT_THRESHOLD = 500


class RTKTriangle(IModel):
    """Triangular RTK unit hydrograph component implementing :class:`~sparsehydro.interfaces.IModel`.

    Constructor arguments seed the :meth:`initialize` parameter registry.
    After :meth:`initialize`, calibratable values live in the scalar-parameter
    registry and are kept in sync with the instance attributes used by the
    physics properties and :func:`triangular_uh`.

    :meth:`predict` convolves a rainfall-excess series with this triangle's
    unit hydrograph kernel and applies the R-scaling, returning the RDII
    contribution for this single component.

    :param R: Fraction of rainfall excess entering the sewer [0, 1].
    :param T: Time to peak [hr] (must be > 0).
    :param K: Ratio of recession time to time-to-peak (must be > 1).

    :raises ValueError: If any parameter is out of range.
    """

    model_name = "rtk-triangle"

    def __init__(
        self,
        R: float = 0.05,
        T: float = 1.0,
        K: float = 1.5,
    ) -> None:
        super().__init__()
        if float(R) < 0.0 or float(R) > 1.0:
            raise ValueError(f"R must be in [0, 1]; got {R!r}")
        if float(T) <= 0.0:
            raise ValueError(f"T must be > 0; got {T!r}")
        if float(K) <= 1.0:
            raise ValueError(f"K must be > 1.0; got {K!r}")
        self.R = float(R)
        self.T = float(T)
        self.K = float(K)
        self._prepared_df: pd.DataFrame | None = None
        self._dt_hours: float = 1.0

    # ------------------------------------------------------------------
    # Physics properties
    # ------------------------------------------------------------------

    @property
    def base_duration(self) -> float:
        """Total triangle duration in hours: T * (1 + K).

        :rtype: float
        """
        return self.T * (1.0 + self.K)

    @property
    def peak_ordinate(self) -> float:
        """Peak ordinate value 2 / (T * (1 + K)), ensuring unit area.

        :rtype: float
        """
        return 2.0 / (self.T * (1.0 + self.K))

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register R, T, K as scalar parameters and advance to INITIALIZED."""
        self.register_scalar_parameter(ScalarParameter(
            "R", value=self.R, lower_bound=0.0, upper_bound=1.0,
            units="-", description="Fraction of rainfall excess entering the sewer",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "T", value=self.T, lower_bound=0.1, upper_bound=240.0,
            units="hr", description="Time to peak",
        ))
        self.register_scalar_parameter(ScalarParameter(
            "K", value=self.K, lower_bound=1.001, upper_bound=10.0,
            units="-", description="Recession-to-peak ratio",
        ))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within bounds.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: pd.DataFrame) -> None:
        """Load rainfall-excess forcing data and infer time step.

        :param data: DataFrame with columns ``datetime`` and ``p_excess_mm``.
        :type data: pandas.DataFrame
        :raises ValueError: If required columns are absent.
        """
        required = {"datetime", "p_excess_mm"}
        missing = required - set(data.columns)
        if missing:
            raise ValueError(
                f"prepare() data is missing required columns: {sorted(missing)}"
            )
        df = data.sort_values("datetime").reset_index(drop=True).copy()
        diffs = df["datetime"].diff().dropna()
        self._dt_hours = pd.Timedelta(diffs.median()).total_seconds() / 3600.0
        self._prepared_df = df
        self._sync_from_params()
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall excess with this triangle's UH kernel and apply R-scaling.

        Parameter values changed after :meth:`prepare` are picked up
        automatically (required by :class:`~sparsehydro.calibration.problem.CalibrationProblem`).

        :returns: DataFrame with columns ``datetime`` and ``rdii_mm``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._prepared_df is None:
            raise RuntimeError("Call prepare(data) before predict().")

        self._sync_from_params()
        p_excess = self._prepared_df["p_excess_mm"].to_numpy(dtype=float)
        kernel = triangular_uh(self, self._dt_hours)
        n = len(p_excess)

        if max(n, len(kernel)) > _FFT_THRESHOLD:
            from scipy.signal import fftconvolve  # type: ignore[import]
            conv = fftconvolve(p_excess, kernel, mode="full")[:n]
        else:
            conv = np.convolve(p_excess, kernel, mode="full")[:n]

        rdii = np.clip(self.R * conv, 0.0, None)
        result = pd.DataFrame({
            "datetime": self._prepared_df["datetime"].values,
            "rdii_mm": rdii,
        })
        self._state = ModelState.PREDICTED
        return result

    def finalize(self) -> None:
        """Release stored forcing data and advance to FINALIZED."""
        self._prepared_df = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _sync_from_params(self) -> None:
        """Update instance attributes from the scalar-parameter registry."""
        self.R = self.get_scalar_parameter("R").value
        self.T = self.get_scalar_parameter("T").value
        self.K = self.get_scalar_parameter("K").value

    def __repr__(self) -> str:
        return f"RTKTriangle(R={self.R!r}, T={self.T!r}, K={self.K!r})"


def _triangle_value(t: float, T: float, K: float) -> float:
    """Evaluate the triangular UH function at continuous time t.

    :param t: Time since pulse [hr] (must be >= 0).
    :param T: Time to peak [hr].
    :param K: Recession ratio.
    :returns: Ordinate value [1/hr].
    """
    base = T * (1.0 + K)
    peak = 2.0 / base
    if t <= 0.0:
        return 0.0
    if t <= T:
        return peak * (t / T)
    if t <= base:
        return peak * (base - t) / (base - T)
    return 0.0


def _integrate_triangle(t0: float, t1: float, T: float, K: float) -> float:
    """Integrate the triangular UH exactly over [t0, t1].

    Handles breakpoints at T and T*(1+K) analytically to avoid
    trapezoidal error at the apex and zero-crossing.

    :param t0: Start of interval [hr].
    :param t1: End of interval [hr].
    :param T: Time to peak [hr].
    :param K: Recession ratio.
    :returns: Integral [dimensionless = mm per mm of excess].
    """
    base = T * (1.0 + K)
    breaks = [t0]
    for bp in (T, base):
        if t0 < bp < t1:
            breaks.append(bp)
    breaks.append(t1)
    breaks.sort()

    total = 0.0
    for a, b in zip(breaks[:-1], breaks[1:]):
        fa = _triangle_value(a, T, K)
        fb = _triangle_value(b, T, K)
        total += 0.5 * (fa + fb) * (b - a)
    return total


def triangular_uh(
    triangle: RTKTriangle,
    dt_hours: float,
    n_steps: int | None = None,
) -> np.ndarray:
    """Compute the unit hydrograph ordinate array for one RTK triangle.

    Each element represents the average flow rate [1/hr] over its interval,
    computed by analytically integrating the piecewise-linear triangle
    (breakpoints at T and T*(1+K) are handled exactly — no trapezoidal error
    at the apex).

    The array satisfies::

        np.sum(ordinates) * dt_hours ≈ 1.0   (within 1e-10 relative error)

    Multiply by ``R * P_excess`` [mm] to obtain RDII depth [mm].

    :param triangle: RTK triangle parameters.
    :type triangle: RTKTriangle
    :param dt_hours: Time-step size [hr].  Must match the rainfall series.
    :type dt_hours: float
    :param n_steps: Number of output steps.  Defaults to
        ``ceil(base_duration / dt_hours) + 1`` (enough to capture the full
        triangle including a trailing zero).
    :type n_steps: int, optional
    :returns: 1-D array of UH ordinates [1/hr], length ``n_steps``.
    :rtype: numpy.ndarray
    """
    if dt_hours <= 0.0:
        raise ValueError(f"dt_hours must be > 0; got {dt_hours!r}")

    base = triangle.base_duration
    if n_steps is None:
        n_steps = math.ceil(base / dt_hours) + 1

    ordinates = np.zeros(n_steps, dtype=float)
    for j in range(n_steps):
        t0 = j * dt_hours
        t1 = t0 + dt_hours
        if t0 >= base:
            break
        ordinates[j] = _integrate_triangle(t0, t1, triangle.T, triangle.K) / dt_hours

    return ordinates
