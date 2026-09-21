"""Shared lifecycle machinery for unit hydrograph components.

Every UH model differs only in :meth:`~IUnitHydroComponent.get_kernel`; the
prepare/predict/finalize lifecycle around it was previously duplicated verbatim
across seven classes.  :class:`UnitHydrographBase` holds that machinery once.

**The predict inversion.**  ``predict_flow()`` is the numeric implementation,
``predict_arrays()`` wraps it in a column mapping, and ``predict()`` is a thin
adapter that builds a DataFrame from ``predict_arrays()``.  Keeping them in that
order -- rather than writing a separate "fast path" beside the DataFrame one --
is what makes it structurally impossible for calibration results to drift away
from notebook results.

**Forcing cache.**  ``prepare()`` materialises the rainfall and datetime columns
into contiguous ndarrays once, so ``predict()`` never touches pandas.  Everything
cached is recomputed on every ``prepare()``; the timestep is reused only when the
caller hands back the *same array object*, which cannot go stale.  The hot-loop
win comes from :meth:`UnitHydrographBase.set_forcing` and from composites passing
a known ``dt_hours`` down, not from guessing that forcing data is unchanged.
"""

from __future__ import annotations

from abc import ABC
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ..base import IUnitHydroComponent
from .kernels import infer_dt_hours_from_values


class UnitHydrographBase(IUnitHydroComponent, ABC):
    """Base class providing the shared UH lifecycle around ``get_kernel()``.

    Subclasses implement only ``__init__``, :meth:`initialize` and
    :meth:`get_kernel`, plus :meth:`_extra_valid` when they carry an ordering
    constraint between parameters.

    :cvar _amplitude_param_name: Name of the scalar parameter scaling the
        kernel; ``"A"`` for every model shipped here.
    :cvar _rain_column: Forcing column read by :meth:`prepare`.
    :cvar _output_column: Name of the predicted-flow column.
    """

    _amplitude_param_name: ClassVar[str | None] = "A"
    _rain_column: ClassVar[str] = "rain"
    _output_column: ClassVar[str] = "Q_pred"
    supports_array_predict: ClassVar[bool] = True

    def __init__(self) -> None:
        super().__init__()
        self._rain: np.ndarray | None = None
        self._datetimes: np.ndarray | None = None
        self._dt_hours: float = 1.0 / 12.0
        self._n: int = 0

    # ------------------------------------------------------------------
    # Parameter access
    # ------------------------------------------------------------------

    def _p(self, name: str) -> float:
        """Return a registered scalar parameter's value.

        Skips :meth:`get_scalar_parameter`'s membership check; ``get_kernel``
        calls this up to six times per evaluation.

        :param name: Scalar parameter name.
        :type name: str
        :returns: Current parameter value.
        :rtype: float
        :raises KeyError: If no such parameter is registered.
        """
        return self._scalar_parameters[name].value

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def validate(self) -> bool:
        """Validate parameter bounds and any subclass ordering constraint.

        :returns: ``True`` if all parameters are in bounds and
            :meth:`_extra_valid` passes.
        :rtype: bool
        """
        ok = self.parameters_valid() and self._extra_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def _extra_valid(self) -> bool:
        """Return whether subclass-specific parameter constraints hold.

        The default imposes none.  :class:`~.models.TriangleUH` and
        :class:`~.peak_tail.PeakTailUH` override it to require ``tp < tt``.

        :returns: ``True`` when no extra constraint is violated.
        :rtype: bool
        """
        return True

    def prepare(self, data: pd.DataFrame, *, dt_hours: float | None = None) -> None:
        """Cache the forcing series as contiguous ndarrays.

        Materialises the rainfall and datetime columns instead of copying the
        whole frame: ``predict()`` only ever reads those two, and with
        PyArrow-backed dtypes the frame copy is the most expensive thing here.

        :param data: DataFrame with ``datetime`` and a rainfall column.
        :type data: pandas.DataFrame
        :param dt_hours: Known timestep size [hr].  When supplied, timestep
            inference is skipped entirely -- composites that already know ``dt``
            should pass it so it is never re-derived in the calibration loop.
        :type dt_hours: float or None
        :returns: Nothing.
        :rtype: None
        :raises KeyError: If the rainfall column is absent.
        """
        rain = np.ascontiguousarray(
            data[self._rain_column].to_numpy(dtype=float)
        )
        datetimes = pd.to_datetime(data["datetime"]).to_numpy()
        self.prepare_arrays(rain, datetimes, dt_hours=dt_hours)

    def prepare_arrays(
        self,
        rain: np.ndarray,
        datetimes: np.ndarray | None = None,
        *,
        dt_hours: float | None = None,
    ) -> None:
        """Cache forcing arrays directly, bypassing pandas entirely.

        :param rain: Rainfall (or rainfall-excess) series.
        :type rain: numpy.ndarray
        :param datetimes: Matching ``datetime64`` array.  Required unless
            *dt_hours* is given and no datetime output is needed.
        :type datetimes: numpy.ndarray or None
        :param dt_hours: Known timestep size [hr]; inferred from *datetimes*
            when omitted.  Reused without re-inference only when *datetimes* is
            the very same array object as the currently cached one.
        :type dt_hours: float or None
        :returns: Nothing.
        :rtype: None
        """
        self._rain = np.ascontiguousarray(rain, dtype=float)
        self._n = int(self._rain.shape[0])

        if dt_hours is not None:
            self._dt_hours = float(dt_hours)

        if datetimes is None:
            self._datetimes = None
        else:
            arr = np.asarray(datetimes)
            # Identity, not equality: only the same object can safely skip
            # re-inference.  Anything else recomputes.
            same_object = self._datetimes is arr
            self._datetimes = arr
            if dt_hours is None and not same_object:
                self._dt_hours = infer_dt_hours_from_values(arr)

        self._state = ModelState.PREPARED

    def set_forcing(self, rain: np.ndarray) -> None:
        """Replace the rainfall forcing, keeping the cached time base and ``dt``.

        Lets a composite swap in a freshly computed rainfall-excess series each
        evaluation without rebuilding a DataFrame or re-inferring the timestep.

        :param rain: New forcing series; must match the prepared length.
        :type rain: numpy.ndarray
        :returns: Nothing.
        :rtype: None
        :raises RuntimeError: If called before :meth:`prepare`.
        :raises ValueError: If *rain* has a different length than the prepared
            series.
        """
        if self._rain is None:
            raise RuntimeError("Call prepare() before set_forcing().")
        arr = np.ascontiguousarray(rain, dtype=float)
        if arr.shape[0] != self._n:
            raise ValueError(
                f"set_forcing() length {arr.shape[0]} does not match the "
                f"prepared series length {self._n}."
            )
        self._rain = arr
        self._state = ModelState.PREPARED

    def predict_flow(self) -> np.ndarray:
        """Convolve the cached forcing with the scaled kernel.

        :returns: Predicted flow array of the prepared length.
        :rtype: numpy.ndarray
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        if self._rain is None:
            raise RuntimeError("Call prepare() before predict().")
        A = self._p(self._amplitude_param_name) if self._amplitude_param_name else 1.0
        dt = self._dt_hours
        kernel = self.get_kernel(dt_hours=dt)
        # Left-to-right exactly as the original `kernel * A * dt`: reassociating
        # to `kernel * (A * dt)` shifts results by ~1e-16 and breaks the pins.
        q = np.convolve(self._rain, kernel * A * dt, mode="full")[: self._n]
        self._state = ModelState.PREDICTED
        return q

    def predict_arrays(self) -> dict[str, np.ndarray]:
        """Return the prediction as plain arrays, without building a DataFrame.

        :returns: Mapping with ``datetime`` (when a time base is cached) and the
            predicted-flow column.
        :rtype: dict[str, numpy.ndarray]
        """
        q = self.predict_flow()
        if self._datetimes is None:
            return {self._output_column: q}
        return {"datetime": self._datetimes, self._output_column: q}

    def predict(self) -> pd.DataFrame:
        """Convolve rainfall with the UH kernel and return predicted flow.

        A thin adapter over :meth:`predict_arrays`, so the DataFrame and array
        paths cannot diverge.

        :returns: DataFrame with columns ``datetime`` and ``Q_pred``.
        :rtype: pandas.DataFrame
        :raises RuntimeError: If :meth:`prepare` has not been called.
        """
        return pd.DataFrame(self.predict_arrays())

    def finalize(self) -> None:
        """Release cached forcing arrays and advance to FINALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._rain = None
        self._datetimes = None
        self._n = 0
        self._state = ModelState.FINALIZED

    # ------------------------------------------------------------------
    # Backwards compatibility
    # ------------------------------------------------------------------

    @property
    def _data(self) -> pd.DataFrame | None:
        """Rebuild the forcing frame that used to be cached verbatim.

        Retained so external code that reached into ``model._data`` keeps
        working.  It is rebuilt on access rather than stored, and nothing on the
        prediction path uses it.

        :returns: Frame with ``datetime`` and the rainfall column, or ``None``
            before :meth:`prepare`.
        :rtype: pandas.DataFrame or None
        """
        if self._rain is None:
            return None
        cols: dict[str, Any] = {}
        if self._datetimes is not None:
            cols["datetime"] = self._datetimes
        cols[self._rain_column] = self._rain
        return pd.DataFrame(cols)


__all__ = ["UnitHydrographBase"]
