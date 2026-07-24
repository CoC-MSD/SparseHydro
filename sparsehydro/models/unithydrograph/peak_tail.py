"""Peak + tail composite unit hydrograph.

Implements the Parsimonious Functions "ensemble" unit hydrograph as a single
:class:`~sparsehydro.models.IUnitHydroComponent`: a fast peak kernel (triangular
by default) blended with a slower tail kernel (gamma by default), sharing one
amplitude ``A``, a blend weight ``w`` (tail area fraction), and a common time
delay ``td``.

Kernel::

    u(t) = A * [ (1 - w) * peak_norm(t - td) + w * tail_norm(t - td) ]

where ``peak_norm`` and ``tail_norm`` are each normalised to unit area, so ``w``
is the fraction of the total UH area carried by the recession tail.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pandas as pd

from ...enums import ModelState
from ..base import IUnitHydroComponent
from ...parameters import ScalarParameter
from .models import _MAX_STEPS, _infer_dt_hours, _normalize_kernel, _trim_pad


def _triangle_raw(tp: float, tt: float, ts: np.ndarray) -> np.ndarray:
    """Return unnormalised triangular ordinates on shifted time base *ts*.

    :param tp: Time to peak in steps.
    :type tp: float
    :param tt: Total duration in steps (``tt > tp``).
    :type tt: float
    :param ts: Shifted time base (``t - td``); values ``<= 0`` yield zero.
    :type ts: numpy.ndarray
    :returns: Non-negative triangular ordinates.
    :rtype: numpy.ndarray
    """
    raw = np.zeros_like(ts)
    rising = (ts > 0) & (ts <= tp)
    raw[rising] = ts[rising] / tp
    falling = (ts > tp) & (ts <= tt)
    raw[falling] = 1.0 - (ts[falling] - tp) / (tt - tp)
    return np.maximum(raw, 0.0)


def _gamma_raw(tt: float, tp: float, ts: np.ndarray) -> np.ndarray:
    """Return unnormalised gamma ordinates on shifted time base *ts*.

    :param tt: Gamma shape parameter.
    :type tt: float
    :param tp: Gamma scale / time-to-peak in steps.
    :type tp: float
    :param ts: Shifted time base (``t - td``); values ``<= 0`` yield zero.
    :type ts: numpy.ndarray
    :returns: Non-negative gamma ordinates.
    :rtype: numpy.ndarray
    """
    raw = np.where(ts > 0.0, (ts / tp) ** tt * np.exp(-ts / tp), 0.0)
    return np.maximum(raw, 0.0)


class PeakTailUH(IUnitHydroComponent):
    """Blended peak (triangle) + tail (gamma) unit hydrograph.

    This is the default sequential-fit model in the Parsimonious workflow: a
    sharp triangular peak captures the rising limb / crest while a gamma tail
    captures the extended recession.

    :param A: Effective area ratio (stormflow / rain volume).  Bounds [0, 1e4].
    :type A: float
    :param w: Tail area fraction in ``[0, 1]``.  Bounds [0, 1].
    :type w: float
    :param td: Shared response delay in time steps.  Bounds [0, 200].
    :type td: float
    :param peak_tp: Peak (triangle) time to peak in steps.  Bounds [2, 500].
    :type peak_tp: float
    :param peak_tt: Peak (triangle) total duration in steps.  Bounds [5, 1000].
        Must be ``> peak_tp``.
    :type peak_tt: float
    :param tail_tt: Tail (gamma) shape parameter.  Bounds [0.01, 50].
    :type tail_tt: float
    :param tail_tp: Tail (gamma) scale / time-to-peak in steps.  Bounds [0.01, 500].
    :type tail_tp: float
    """

    model_name: ClassVar[str] = "peak-tail-uh"
    _amplitude_param_name: ClassVar[str | None] = "A"

    def __init__(
        self,
        A: float = 100.0,
        w: float = 0.5,
        td: float = 5.0,
        peak_tp: float = 20.0,
        peak_tt: float = 50.0,
        tail_tt: float = 2.0,
        tail_tp: float = 5.0,
    ) -> None:
        super().__init__()
        self._A_init = float(A)
        self._w_init = float(w)
        self._td_init = float(td)
        self._peak_tp_init = float(peak_tp)
        self._peak_tt_init = float(peak_tt)
        self._tail_tt_init = float(tail_tt)
        self._tail_tp_init = float(tail_tp)
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        """Register the blend, delay, peak, and tail parameters; advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self._scalars = {}
        self._vectors = {}
        self._constraints = []
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("w", value=self._w_init, lower_bound=0.0, upper_bound=1.0, description="Tail area fraction"))
        self.register_scalar_parameter(ScalarParameter("td", value=self._td_init, lower_bound=0.0, upper_bound=200.0, units="steps", description="Shared response delay in time steps"))
        self.register_scalar_parameter(ScalarParameter("peak_tp", value=self._peak_tp_init, lower_bound=2.0, upper_bound=500.0, units="steps", description="Peak triangle time to peak"))
        self.register_scalar_parameter(ScalarParameter("peak_tt", value=self._peak_tt_init, lower_bound=5.0, upper_bound=1000.0, units="steps", description="Peak triangle total duration"))
        self.register_scalar_parameter(ScalarParameter("tail_tt", value=self._tail_tt_init, lower_bound=0.01, upper_bound=50.0, description="Tail gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tail_tp", value=self._tail_tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Tail gamma time to peak"))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        """Validate parameter bounds and the ``peak_tp < peak_tt`` ordering constraint.

        :returns: ``True`` if all parameters are within bounds and ``peak_tp < peak_tt``.
        :rtype: bool
        """
        ok = self.parameters_valid()
        if ok and self.get_scalar_parameter("peak_tp").value >= self.get_scalar_parameter("peak_tt").value:
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
        """Convolve rainfall with the blended peak+tail kernel and return predicted flow.

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
        """Return the normalized blended peak+tail UH ordinate array.

        Both the peak and tail kernels are normalised to unit area before
        blending, so the blend weight ``w`` is the tail's fraction of the total
        UH area and the combined kernel again satisfies ``sum * dt_hours ≈ 1``.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Normalized UH ordinates [1/hr] such that ``sum * dt_hours ≈ 1``.
        :rtype: numpy.ndarray
        """
        w = min(max(self.get_scalar_parameter("w").value, 0.0), 1.0)
        td = max(self.get_scalar_parameter("td").value, 0.0)
        peak_tp = max(self.get_scalar_parameter("peak_tp").value, 1e-6)
        peak_tt = max(self.get_scalar_parameter("peak_tt").value, peak_tp + 1e-6)
        tail_tt = self.get_scalar_parameter("tail_tt").value
        tail_tp = max(self.get_scalar_parameter("tail_tp").value, 1e-6)

        max_steps = min(_MAX_STEPS, max(int(peak_tt + 5 * tail_tp + td + 1), 20))
        t = np.arange(max_steps, dtype=float)
        ts = t - td

        peak_norm = _normalize_kernel(_triangle_raw(peak_tp, peak_tt, ts), dt_hours)
        tail_norm = _normalize_kernel(_gamma_raw(tail_tt, tail_tp, ts), dt_hours)
        blend = (1.0 - w) * peak_norm + w * tail_norm
        return _trim_pad(blend, n_steps)


__all__ = ["PeakTailUH"]
