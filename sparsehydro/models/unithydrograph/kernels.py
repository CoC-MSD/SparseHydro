"""Shared unit-hydrograph kernel helpers.

Every UH model builds its ordinates the same way: evaluate a closed-form shape
on a time base expressed in **time steps**, then normalise so the discrete
ordinates carry unit volume::

    sum(get_kernel(dt)) * dt ~ 1.0        [kernel units: 1/hr]

Because :meth:`predict` multiplies the kernel by ``dt`` again, the two ``dt``
factors cancel exactly and predicted flow is independent of the timestep --
the normalisation is what makes convolution volume-conserving, not a unit
conversion.

These helpers previously lived as private names in :mod:`.models`; they are
re-exported there under their old ``_``-prefixed names so existing imports keep
working.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: Maximum kernel length in time steps (72 h at 5-minute intervals).
MAX_STEPS: int = 864


def infer_dt_hours(data: pd.DataFrame) -> float:
    """Infer the timestep size from a frame's ``datetime`` column.

    :param data: DataFrame containing a ``datetime`` column.
    :type data: pandas.DataFrame
    :returns: Median timestep size [hr]; ``1/12`` (5 min) for fewer than 2 rows.
    :rtype: float
    """
    if len(data) < 2:
        return 1.0 / 12.0
    dt = data["datetime"].diff().median()
    return dt.total_seconds() / 3600.0


def infer_dt_hours_from_values(values: np.ndarray) -> float:
    """Infer the timestep size from a ``datetime64`` array.

    The ndarray equivalent of :func:`infer_dt_hours`, avoiding the pandas
    ``Series.diff().median()`` round-trip on the calibration hot path.

    The final division is written as ``(ns / 1e9) / 3600.0`` rather than the
    tidier ``ns / 3.6e12`` so it reproduces ``Timedelta.total_seconds() / 3600``
    to the last bit -- the two spellings differ by an ulp for some inputs, and
    ``dt_hours`` scales every kernel ordinate.

    :param values: 1-D ``datetime64`` array.
    :type values: numpy.ndarray
    :returns: Median timestep size [hr]; ``1/12`` (5 min) for fewer than 2
        elements or a non-positive median.
    :rtype: float
    """
    if values is None or len(values) < 2:
        return 1.0 / 12.0
    diffs = np.diff(np.asarray(values, dtype="datetime64[ns]").view("i8"))
    if diffs.size == 0:
        return 1.0 / 12.0
    median_ns = float(np.median(diffs))
    if not np.isfinite(median_ns) or median_ns <= 0.0:
        return 1.0 / 12.0
    return (median_ns / 1e9) / 3600.0


def normalize_kernel(raw: np.ndarray, dt_hours: float) -> np.ndarray:
    """Normalise ordinates so ``sum(result) * dt_hours ~ 1.0``.

    :param raw: Unnormalised kernel ordinates.
    :type raw: numpy.ndarray
    :param dt_hours: Timestep size [hr].
    :type dt_hours: float
    :returns: Normalised kernel [1/hr]; all zeros if the raw sum is non-positive.
    :rtype: numpy.ndarray
    """
    total = float(np.sum(raw))
    if total <= 0.0:
        return np.zeros_like(raw)
    return raw / (total * dt_hours)


def trim_pad(arr: np.ndarray, n_steps: int | None) -> np.ndarray:
    """Trim or zero-pad *arr* to exactly *n_steps* elements.

    :param arr: Input array.
    :type arr: numpy.ndarray
    :param n_steps: Target length, or ``None`` to leave *arr* unchanged.
    :type n_steps: int | None
    :returns: Array of length *n_steps* (or *arr* unchanged when *n_steps* is
        ``None``).
    :rtype: numpy.ndarray
    """
    if n_steps is None:
        return arr
    if len(arr) >= n_steps:
        return arr[:n_steps]
    return np.pad(arr, (0, n_steps - len(arr)))


__all__ = [
    "MAX_STEPS",
    "infer_dt_hours",
    "infer_dt_hours_from_values",
    "normalize_kernel",
    "trim_pad",
]
