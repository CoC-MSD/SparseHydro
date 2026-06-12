"""SeasonalityModel — deterministic Fourier-series sanitary base-flow estimator.

Models the cyclic sanitary (dry-weather) component of sewer flow as a sum of
sine and cosine harmonics over one or more time-domain periods.  The model
mirrors the structure of the PyTorch ``Seasonality`` module but is implemented
entirely in NumPy so it integrates with the sparsehydro
:class:`~sparsehydro.interfaces.IModel` lifecycle and can be calibrated by any
:class:`~sparsehydro.calibration.CalibrationProblem` / solver combination.

Formula
-------

.. math::

   \\hat{q}[t] = \\sum_p \\sum_{n=1}^{N}
       \\left[ a_{p,n} \\cos\\!\\left(\\frac{2\\pi n\\, f_p[t]}{P_p}\\right)
             + b_{p,n} \\sin\\!\\left(\\frac{2\\pi n\\, f_p[t]}{P_p}\\right)
       \\right]

where:

* *p* — period index (one per configured time feature),
* *f_p[t]* — time-feature value at step *t* (e.g. hour-of-day),
* *P_p* — period of that feature (e.g. 24.0 for hour-of-day),
* *N* = ``n_terms`` — configurable Fourier order (number of harmonics).

Usage::

    from sparsehydro.rdii.seasonality import SeasonalityModel

    model = SeasonalityModel(
        periods={
            "hour_of_day": 24.0,
            "day_of_week": 7.0,
            "day_of_year": 365.25,
        },
        n_terms=5,
        output_name="sanitary_cfs",
        coeff_bounds=(-50.0, 50.0),
    )
    model.initialize()
    model.validate()
    model.prepare(df)         # df must have "datetime"; time features auto-computed
    result = model.predict()  # DataFrame: datetime, sanitary_cfs
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IModel
from ..parameters import FieldRecord, ScalarParameter
from ..registry import registry

# ---------------------------------------------------------------------------
# Auto-computed feature keys and their period defaults
# ---------------------------------------------------------------------------

_DEFAULT_PERIODS: dict[str, float] = {
    "hour_of_day": 24.0,
    "day_of_week": 7.0,
    "day_of_year": 365.25,
}

_AUTO_FEATURES = frozenset(_DEFAULT_PERIODS)


def _compute_auto_feature(key: str, dt: pd.Series) -> np.ndarray:
    """Return the time-feature array for a standard period key."""
    frac_day = dt.dt.hour / 24.0 + dt.dt.minute / 1440.0 + dt.dt.second / 86400.0
    if key == "hour_of_day":
        return (dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0).to_numpy(dtype=float)
    if key == "day_of_week":
        return (dt.dt.dayofweek.astype(float) + frac_day).to_numpy(dtype=float)
    if key == "day_of_year":
        return (dt.dt.dayofyear.astype(float) + frac_day).to_numpy(dtype=float)
    raise KeyError(f"No auto-computation rule for feature '{key}'.")  # pragma: no cover


def compute_time_features(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
) -> pd.DataFrame:
    """Return a copy of *df* with floating-point time-feature columns added.

    Three columns are added at full fractional resolution:

    * ``hour_of_day``  = ``hour + minute/60 + second/3600``
    * ``day_of_week``  = ``dayofweek + hour/24 + minute/1440 + second/86400``  (Mon = 0)
    * ``day_of_year``  = ``dayofyear  + hour/24 + minute/1440 + second/86400``

    Pre-existing columns with the same names are overwritten in the copy.

    :param df: Input DataFrame containing a datetime-like column.
    :param datetime_col: Name of the datetime column.  Defaults to ``"datetime"``.
    :returns: A copy of *df* with the three new columns appended.
    :rtype: pandas.DataFrame
    """
    dt = pd.to_datetime(df[datetime_col])
    out = df.copy()
    frac_day = dt.dt.hour / 24.0 + dt.dt.minute / 1440.0 + dt.dt.second / 86400.0
    out["hour_of_day"] = (dt.dt.hour + dt.dt.minute / 60.0 + dt.dt.second / 3600.0).to_numpy(dtype=float)
    out["day_of_week"] = (dt.dt.dayofweek.astype(float) + frac_day).to_numpy(dtype=float)
    out["day_of_year"] = (dt.dt.dayofyear.astype(float) + frac_day).to_numpy(dtype=float)
    return out


@registry.register
class SeasonalityModel(IModel):
    """Fourier-series sanitary base-flow model.

    Estimates the cyclic (dry-weather) component of sewer flow using a truncated
    Fourier series over configurable time periods.  The cosine and sine
    coefficients (``a_{p}_{n}`` and ``b_{p}_{n}``) are
    :class:`~sparsehydro.parameters.ScalarParameter` objects fully exposed to
    the calibration framework.

    :param periods: Mapping from time-feature column name to its period value.
        Three standard feature keys are recognised and **auto-computed from
        the** ``datetime`` **column** when absent from the input DataFrame:

        ``"hour_of_day"`` (period 24.0), ``"day_of_week"`` (7.0),
        ``"day_of_year"`` (365.25).

        Any other key must be present as a column in the DataFrame passed to
        :meth:`prepare`.  Defaults to all three standard features.
    :param n_terms: Number of Fourier harmonics *N* per period (≥ 1).
    :param output_name: Column name for the model output in :meth:`predict`
        results.  Defaults to ``"sanitary_cfs"``.
    :param coeff_bounds: ``(lower, upper)`` bounds applied to every ``a`` and
        ``b`` coefficient.  Defaults to ``(-50.0, 50.0)``.

    Total calibratable parameters: ``2 × len(periods) × n_terms``.

    Example::

        model = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=3)
        model.initialize()
        model.validate()
        model.prepare(df)
        out = model.predict()   # columns: datetime, sanitary_cfs
    """

    model_name = "seasonality"

    def __init__(
        self,
        periods: dict[str, float] | None = None,
        n_terms: int = 5,
        output_name: str = "sanitary_cfs",
        coeff_bounds: tuple[float, float] = (-50.0, 50.0),
    ) -> None:
        super().__init__()
        if n_terms < 1:
            raise ValueError(f"n_terms must be >= 1; got {n_terms!r}.")
        lb, ub = coeff_bounds
        if lb > ub:
            raise ValueError(
                f"coeff_bounds lower ({lb}) must be <= upper ({ub})."
            )

        self._periods: dict[str, float] = dict(periods) if periods is not None else dict(_DEFAULT_PERIODS)
        self._n_terms: int = n_terms
        self._output_name: str = output_name
        self._coeff_bounds: tuple[float, float] = coeff_bounds

        self._features: dict[str, np.ndarray] = {}
        self._datetime: np.ndarray | None = None

    # ------------------------------------------------------------------
    # IModel lifecycle
    # ------------------------------------------------------------------

    def initialize(self) -> None:
        """Register all Fourier coefficient parameters and advance to INITIALIZED.

        Registers ``a_{p}_{n}`` (cosine) and ``b_{p}_{n}`` (sine) for every
        period key *p* and harmonic index *n* in ``1 … n_terms``.
        """
        lb, ub = self._coeff_bounds
        for p in self._periods:
            for n in range(1, self._n_terms + 1):
                self.register_scalar_parameter(ScalarParameter(
                    name=f"a_{p}_{n}",
                    value=0.0,
                    lower_bound=lb,
                    upper_bound=ub,
                    units="-",
                    description=f"Cosine coefficient: period '{p}', harmonic {n}",
                ))
                self.register_scalar_parameter(ScalarParameter(
                    name=f"b_{p}_{n}",
                    value=0.0,
                    lower_bound=lb,
                    upper_bound=ub,
                    units="-",
                    description=f"Sine coefficient: period '{p}', harmonic {n}",
                ))
        # --- Output field metadata ---
        self.register_output_field(FieldRecord(
            name="datetime",
            description="Simulation time step",
        ))
        self.register_output_field(FieldRecord(
            name=self._output_name,
            units="",
            description=(
                f"Fourier-series sanitary base flow — "
                f"{len(self._periods)} period(s) × {self._n_terms} harmonics"
            ),
            calibratable=True,
        ))

        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and advance to VALIDATED.

        :returns: ``True`` if all parameters are within their bounds.
        :rtype: bool
        """
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: Any, **kwargs: Any) -> None:
        """Extract and cache time features from *data*.

        :param data: A :class:`pandas.DataFrame` containing at minimum a
            ``datetime`` column.  Standard feature columns
            (``hour_of_day``, ``day_of_week``, ``day_of_year``) are computed
            automatically from ``datetime`` when absent.  Any other period key
            must be present as a numeric column.
        :raises ValueError: If a required non-standard feature column is absent.
        :raises KeyError: If the ``datetime`` column is missing.
        """
        df = data
        dt: pd.Series = pd.to_datetime(df["datetime"])

        features: dict[str, np.ndarray] = {}
        for key in self._periods:
            if key in _AUTO_FEATURES and key not in df.columns:
                features[key] = _compute_auto_feature(key, dt)
            elif key in df.columns:
                features[key] = df[key].to_numpy(dtype=float)
            else:
                raise ValueError(
                    f"SeasonalityModel: feature column '{key}' is not a standard "
                    f"auto-computed key and is not present in the input DataFrame.  "
                    f"Standard auto-computed keys: {sorted(_AUTO_FEATURES)}."
                )

        self._features = features
        self._datetime = dt.to_numpy()
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        """Evaluate the Fourier series and return the sanitary flow estimate.

        :returns: DataFrame with columns ``datetime`` and ``{output_name}``.
        :rtype: pandas.DataFrame
        """
        n_steps = len(self._datetime)
        output = np.zeros(n_steps, dtype=float)

        for p, period in self._periods.items():
            f = self._features[p]
            for n in range(1, self._n_terms + 1):
                phase = 2.0 * np.pi * n * f / period
                a = self.get_scalar_parameter(f"a_{p}_{n}").value
                b = self.get_scalar_parameter(f"b_{p}_{n}").value
                output += a * np.cos(phase) + b * np.sin(phase)

        self._state = ModelState.PREDICTED
        return pd.DataFrame({"datetime": self._datetime, self._output_name: output})

    def finalize(self) -> None:
        """Release cached data and advance to FINALIZED."""
        self._features = {}
        self._datetime = None
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    @property
    def periods(self) -> dict[str, float]:
        """Configured period mapping ``{feature_key: period_value}``."""
        return dict(self._periods)

    @property
    def n_terms(self) -> int:
        """Number of Fourier harmonics per period."""
        return self._n_terms

    @property
    def output_name(self) -> str:
        """Column name for the combined output in :meth:`predict` results."""
        return self._output_name
