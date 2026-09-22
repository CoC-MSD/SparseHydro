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

from ...enums import ModelState
from ...parameters import ScalarParameter
from .base import UnitHydrographBase
from .kernels import (
    MAX_STEPS as _MAX_STEPS,
    finalize_kernel,
    kernel_errstate,
    normalize_kernel as _normalize_kernel,
    trim_pad as _trim_pad,
)


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
    with kernel_errstate():
        raw = np.where(ts > 0.0, (ts / tp) ** tt * np.exp(-ts / tp), 0.0)
        return np.maximum(raw, 0.0)


class PeakTailUH(UnitHydrographBase):
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

    def initialize(self) -> None:
        """Register the blend, delay, peak, and tail parameters; advance to INITIALIZED.

        :returns: Nothing.
        :rtype: None
        """
        self.register_scalar_parameter(ScalarParameter("A", value=self._A_init, lower_bound=0.0, upper_bound=1e4, description="Effective area ratio"))
        self.register_scalar_parameter(ScalarParameter("w", value=self._w_init, lower_bound=0.0, upper_bound=1.0, description="Tail area fraction"))
        self.register_scalar_parameter(ScalarParameter("td", value=self._td_init, lower_bound=0.0, upper_bound=200.0, units="steps", description="Shared response delay in time steps"))
        self.register_scalar_parameter(ScalarParameter("peak_tp", value=self._peak_tp_init, lower_bound=2.0, upper_bound=500.0, units="steps", description="Peak triangle time to peak"))
        self.register_scalar_parameter(ScalarParameter("peak_tt", value=self._peak_tt_init, lower_bound=5.0, upper_bound=1000.0, units="steps", description="Peak triangle total duration"))
        self.register_scalar_parameter(ScalarParameter("tail_tt", value=self._tail_tt_init, lower_bound=0.01, upper_bound=50.0, description="Tail gamma shape parameter"))
        self.register_scalar_parameter(ScalarParameter("tail_tp", value=self._tail_tp_init, lower_bound=0.01, upper_bound=500.0, units="steps", description="Tail gamma time to peak"))
        self._state = ModelState.INITIALIZED

    def _extra_valid(self) -> bool:
        """Return whether the ``peak_tp < peak_tt`` ordering constraint holds.

        :returns: ``True`` when the triangle's time-to-peak precedes its duration.
        :rtype: bool
        """
        return self._p("peak_tp") < self._p("peak_tt")

    def _weighted_parts(self, dt_hours: float) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(1-w)``-scaled peak and ``w``-scaled tail contributions.

        Both share one time base aligned at the common delay, so they overlap and
        sum to the combined kernel.  Factored out so :meth:`get_kernel` and
        :meth:`component_kernels` cannot drift apart.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :returns: ``(peak_contribution, tail_contribution)``, equal length.
        :rtype: tuple[numpy.ndarray, numpy.ndarray]
        """
        w = min(max(self._p("w"), 0.0), 1.0)
        td = max(self._p("td"), 0.0)
        peak_tp = max(self._p("peak_tp"), 1e-6)
        peak_tt = max(self._p("peak_tt"), peak_tp + 1e-6)
        tail_tt = self._p("tail_tt")
        tail_tp = max(self._p("tail_tp"), 1e-6)

        max_steps = min(_MAX_STEPS, max(int(peak_tt + 5 * tail_tp + td + 1), 20))
        t = np.arange(max_steps, dtype=float)
        ts = t - td

        # Each component is sanitised and normalised on its own, then blended.
        # Normalising the *blend* instead would be a no-op mathematically -- two
        # unit-area kernels weighted (1-w) and w already sum to unit area -- but
        # would perturb the result at float level for no benefit.
        #
        # Sanitising matters here too: _gamma_raw evaluates (ts/tp) ** tt, which
        # overflows to inf for small tp and large tt, and inf * exp(-ts/tp) is
        # NaN.  Without this the whole kernel turns to NaN.
        peak_norm = finalize_kernel(_triangle_raw(peak_tp, peak_tt, ts), dt_hours)
        tail_norm = finalize_kernel(_gamma_raw(tail_tt, tail_tp, ts), dt_hours)
        return (1.0 - w) * peak_norm, w * tail_norm

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
        peak_c, tail_c = self._weighted_parts(dt_hours)
        return _trim_pad(peak_c + tail_c, n_steps)

    def component_kernels(self, dt_hours: float, n_steps: int | None = None) -> dict[str, np.ndarray]:
        """Return the weighted peak and tail contributions and their sum.

        All three arrays share the same time base (aligned at the common delay),
        so the peak and tail contributions **overlap** and add up to the combined
        kernel: ``peak + tail == combined``.  Useful for overlaying the two
        constituent shapes on one axis rather than seeing only the blended curve.

        :param dt_hours: Time-step size [hr].
        :type dt_hours: float
        :param n_steps: Number of output steps; defaults to the natural support.
        :type n_steps: int | None
        :returns: Mapping with keys ``"peak"``, ``"tail"``, ``"combined"`` — each
            a normalized ordinate array (the peak/tail entries are already scaled
            by ``1 - w`` and ``w`` respectively).
        :rtype: dict[str, numpy.ndarray]
        """
        peak_c, tail_c = self._weighted_parts(dt_hours)
        return {
            "peak": _trim_pad(peak_c, n_steps),
            "tail": _trim_pad(tail_c, n_steps),
            "combined": _trim_pad(peak_c + tail_c, n_steps),
        }


__all__ = ["PeakTailUH"]
