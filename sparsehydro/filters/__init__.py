"""sparsehydro.filters — Savitzky-Golay filtering for stormflow preprocessing.

Provides a self-contained pipeline for smoothing raw stormflow and computing
derivative signals used by the event-detection algorithm in
:mod:`sparsehydro.events`.

Public API
----------
- :class:`FilterResult` — dataclass holding the four signal arrays + thresholds
- :func:`apply_savgol_filter` — computes sg_0, sg_1, sg_2 from a rain/stormflow DataFrame
- :func:`compute_thresholds` — attaches percentile-based detection thresholds

Quick start::

    from sparsehydro.filters import apply_savgol_filter, compute_thresholds

    result = apply_savgol_filter(rain_stormflow_df, window_length=48)
    result = compute_thresholds(result, sg_0_per=0.15, sg_1_per=0.25, sg_2_per=0.25)
    print(result.thresholds)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


@dataclass
class FilterResult:
    """Output of :func:`apply_savgol_filter`.

    Attributes
    ----------
    datetime : pd.Series
        Timestamps aligned to the stormflow input.
    raw_flow : np.ndarray
        Original stormflow values (clipped to ≥ 0).
    sg_0 : np.ndarray
        Smoothed signal (Savitzky-Golay 0th-order output).
    sg_1 : np.ndarray
        First derivative of the smoothed signal (slope).
    sg_2 : np.ndarray
        Second derivative of the smoothed signal (curvature).
    thresholds : dict[str, float]
        Populated by :func:`compute_thresholds`.
        Keys: ``"sg_0_th"``, ``"sg_1_th"``, ``"sg_2_th"``.
    """

    datetime: pd.Series
    raw_flow: np.ndarray
    sg_0: np.ndarray
    sg_1: np.ndarray
    sg_2: np.ndarray
    thresholds: dict[str, float] = field(default_factory=dict)

    def to_dataframe(self) -> pd.DataFrame:
        """Return a DataFrame with columns datetime, raw_flow, sg_0, sg_1, sg_2."""
        return pd.DataFrame(
            {
                "datetime": self.datetime,
                "raw_flow": self.raw_flow,
                "sg_0": self.sg_0,
                "sg_1": self.sg_1,
                "sg_2": self.sg_2,
            }
        )


def _get_safe_window(window: float, n: int) -> int:
    """Round *window* to nearest odd integer, capped at *n*, minimum 3."""
    val = int(round(window))
    val = min(val, n)
    val = max(val, 3)
    if val % 2 == 0:
        val = max(3, val - 1)
    return val


def apply_savgol_filter(
    rain_stormflow: pd.DataFrame,
    window_length: int = 48,
    polyorder: int = 2,
    deriv_factor: float = 1.2,
) -> FilterResult:
    """Apply Savitzky-Golay smoothing and compute derivative signals.

    Parameters
    ----------
    rain_stormflow : pd.DataFrame
        Must contain columns ``datetime``, ``rain``, ``stormflow``.
    window_length : int
        Base window for the sg_0 smoother.  Windows for sg_1 and sg_2 are
        scaled by successive powers of *deriv_factor*.
    polyorder : int
        Polynomial order for all filter passes.
    deriv_factor : float
        Multiplier applied to the window length for each derivative pass.

    Returns
    -------
    FilterResult
        Arrays of the same length as *rain_stormflow*; ``thresholds`` is empty
        until :func:`compute_thresholds` is called.
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    stormflow = df["stormflow"].values.astype(float)
    stormflow = np.maximum(stormflow, 0.0)
    n = len(stormflow)

    min_allowed = polyorder + 1 if (polyorder + 1) % 2 != 0 else polyorder + 2

    if n < min_allowed:
        return FilterResult(
            datetime=df["datetime"].reset_index(drop=True),
            raw_flow=stormflow,
            sg_0=stormflow.copy(),
            sg_1=np.zeros(n),
            sg_2=np.zeros(n),
        )

    w0 = _get_safe_window(window_length, n)
    w1 = _get_safe_window(window_length * deriv_factor, n)
    w2 = _get_safe_window(window_length * (deriv_factor**2), n)

    sg_0 = savgol_filter(stormflow, window_length=w0, polyorder=polyorder, mode="interp")
    sg_0 = np.maximum(sg_0, 0.0)
    sg_1 = savgol_filter(sg_0, window_length=w1, polyorder=polyorder, deriv=1, mode="interp")
    sg_2 = savgol_filter(sg_0, window_length=w2, polyorder=polyorder, deriv=2, mode="interp")

    return FilterResult(
        datetime=df["datetime"].reset_index(drop=True),
        raw_flow=stormflow,
        sg_0=sg_0,
        sg_1=sg_1,
        sg_2=sg_2,
    )


def compute_thresholds(
    result: FilterResult,
    sg_0_per: float = 0.15,
    sg_1_per: float = 0.25,
    sg_2_per: float = 0.25,
) -> FilterResult:
    """Attach percentile-based detection thresholds to *result*.

    Parameters
    ----------
    result : FilterResult
        Output of :func:`apply_savgol_filter`.
    sg_0_per : float
        ``sg_0_th = 95th-percentile(sg_0[sg_0 > 0]) * sg_0_per``.
    sg_1_per : float
        ``sg_1_th = std(sg_1) * sg_1_per``.
    sg_2_per : float
        ``sg_2_th = std(sg_2) * sg_2_per``.

    Returns
    -------
    FilterResult
        New :class:`FilterResult` with ``thresholds`` populated (immutable
        update — the original is not modified).
    """
    sg_0_positive = result.sg_0[result.sg_0 > 0]
    if len(sg_0_positive) == 0:
        sg_0_th = 0.0
    else:
        sg_0_th = float(np.percentile(sg_0_positive, 95) * sg_0_per)

    sg_1_th = float(np.std(result.sg_1) * sg_1_per)
    sg_2_th = float(np.std(result.sg_2) * sg_2_per)

    return FilterResult(
        datetime=result.datetime,
        raw_flow=result.raw_flow,
        sg_0=result.sg_0,
        sg_1=result.sg_1,
        sg_2=result.sg_2,
        thresholds={
            "sg_0_th": sg_0_th,
            "sg_1_th": sg_1_th,
            "sg_2_th": sg_2_th,
        },
    )


__all__ = ["FilterResult", "apply_savgol_filter", "compute_thresholds"]
