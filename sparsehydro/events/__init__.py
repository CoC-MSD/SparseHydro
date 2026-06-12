"""sparsehydro.events — Storm event detection and record structures.

Provides derivative-based segmentation of stormflow timeseries into discrete
storm events using Savitzky-Golay signals from :mod:`sparsehydro.filters`.

Public API
----------
- :class:`EventRecord` — dataclass for a single storm event
- :func:`detect_events` — automatic event detection from rain/stormflow data
- :func:`events_to_dataframe` — convert a list of EventRecord to a DataFrame
- :func:`load_events_from_csv` — load pre-defined events from a CSV file

Quick start::

    from sparsehydro.events import detect_events, load_events_from_csv

    events, filter_result = detect_events(rain_stormflow_df, verbose=True)
    # or
    events = load_events_from_csv("events_list.csv")
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter

from ..filters import FilterResult, apply_savgol_filter, compute_thresholds


@dataclass
class EventRecord:
    """A single detected or imported storm event.

    Attributes
    ----------
    event_id : int
        Sequential identifier starting at 1.
    start_datetime : pd.Timestamp
    end_datetime : pd.Timestamp
    peak_datetime : pd.Timestamp
        Timestamp of the sg_0 peak within the event window.
    total_rain : float
        Cumulative rain depth over [start, end].
    total_flow : float
        Cumulative stormflow over [start, end] (clipped to ≥ 0).
    effective_area : float
        ``total_flow / total_rain``.  Zero when ``total_rain == 0``.
    peak_flow : float
        Maximum stormflow value within the event window.
    b2b_start : bool
        True if the event starts at a back-to-back trough.
    b2b_end : bool
        True if the event ends at a back-to-back trough.
    """

    event_id: int
    start_datetime: pd.Timestamp
    end_datetime: pd.Timestamp
    peak_datetime: pd.Timestamp
    total_rain: float
    total_flow: float
    effective_area: float
    peak_flow: float
    b2b_start: bool = False
    b2b_end: bool = False

    def duration_hours(self) -> float:
        """Return event duration in hours."""
        return (self.end_datetime - self.start_datetime).total_seconds() / 3600.0

    def to_dict(self) -> dict:
        """Serialize to plain dict."""
        return {
            "event_id": self.event_id,
            "start_datetime": self.start_datetime,
            "end_datetime": self.end_datetime,
            "peak_datetime": self.peak_datetime,
            "total_rain": self.total_rain,
            "total_flow": self.total_flow,
            "effective_area": self.effective_area,
            "peak_flow": self.peak_flow,
            "b2b_start": self.b2b_start,
            "b2b_end": self.b2b_end,
        }


def events_to_dataframe(events: list[EventRecord]) -> pd.DataFrame:
    """Convert a list of :class:`EventRecord` objects to a DataFrame."""
    if not events:
        return pd.DataFrame(
            columns=[
                "event_id", "start_datetime", "end_datetime", "peak_datetime",
                "total_rain", "total_flow", "effective_area", "peak_flow",
                "b2b_start", "b2b_end",
            ]
        )
    return pd.DataFrame([e.to_dict() for e in events])


def load_events_from_csv(path: str | os.PathLike) -> list[EventRecord]:
    """Load an ``events_list.csv`` as a list of :class:`EventRecord` objects.

    The CSV must have columns ``event_id``, ``start_date``, ``end_date``.
    Fields not present in the CSV are filled with sentinel values.

    Parameters
    ----------
    path : str or path-like

    Returns
    -------
    list[EventRecord]
    """
    df = pd.read_csv(path, parse_dates=["start_date", "end_date"])
    events: list[EventRecord] = []
    for _, row in df.iterrows():
        events.append(
            EventRecord(
                event_id=int(row["event_id"]),
                start_datetime=pd.Timestamp(row["start_date"]),
                end_datetime=pd.Timestamp(row["end_date"]),
                peak_datetime=pd.NaT,
                total_rain=0.0,
                total_flow=0.0,
                effective_area=0.0,
                peak_flow=0.0,
                b2b_start=False,
                b2b_end=False,
            )
        )
    return events


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_peaks_with_thresholds(
    sg_0: np.ndarray,
    height_perc: float,
    prom_perc: float,
    width_perc: float,
    distance: int,
) -> np.ndarray:
    idx_all, props_all = find_peaks(sg_0, height=0, prominence=0, width=1)
    if len(idx_all) == 0:
        return np.array([], dtype=int)

    height_th = np.percentile(props_all["peak_heights"], height_perc)
    prom_th = np.percentile(props_all["prominences"], prom_perc)
    width_th = np.percentile(props_all["widths"], width_perc)

    peaks_idx, _ = find_peaks(
        sg_0, height=height_th, prominence=prom_th, width=width_th, distance=distance,
    )
    return peaks_idx


def _calc_window(window_size, sg_0_win, sg_1_win, sg_0_th, sg_1_th):
    mean_sg0 = float(np.mean(sg_0_win))
    mean_abs_sg1 = float(np.mean(np.abs(sg_1_win)))
    if mean_abs_sg1 > 1e-6 and mean_sg0 > 1e-6:
        calc = int(window_size * (sg_0_th / mean_sg0) * (sg_1_th / mean_abs_sg1))
    else:
        calc = window_size
    return max(2, min(calc, window_size))


def _backward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th):
    lower_bound = min(window_size, peak)
    start_idx = peak
    min_val_bwd = sg_0[peak]
    min_idx_bwd = peak
    current_window_size = window_size

    for i in range(peak - 1, lower_bound - 1, -1):
        curr_val = sg_0[i]
        if curr_val < min_val_bwd:
            min_val_bwd = curr_val
            min_idx_bwd = i

        lookback_start = max(0, i - (current_window_size - 1))
        win_idx = range(lookback_start, i + 1)
        sg0_w, sg1_w, sg2_w = sg_0[win_idx], sg_1[win_idx], sg_2[win_idx]

        sscond = (
            (sg0_w < sg_0_th).all()
            and (np.abs(sg1_w) < sg_1_th).all()
            and (np.abs(sg2_w) < sg_2_th).all()
        )
        if sscond:
            return i, False

        bbcond = (sg0_w > (min_val_bwd + sg_0_th * 2)).all() and (sg1_w < -sg_1_th).all()
        if bbcond:
            return min_idx_bwd, True

        start_idx = i
        current_window_size = _calc_window(window_size, sg0_w, sg1_w, sg_0_th, sg_1_th)

    return start_idx, False


def _forward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th):
    n = len(sg_0)
    upper_bound = max(peak + 1, n - window_size)
    end_idx = peak
    min_val_fwd = sg_0[peak]
    min_idx_fwd = peak
    current_window_size = window_size

    for i in range(peak + 1, upper_bound):
        curr_val = sg_0[i]
        if curr_val < min_val_fwd:
            min_val_fwd = curr_val
            min_idx_fwd = i

        lookforward_end = min(upper_bound, i + current_window_size)
        win_idx = range(i, lookforward_end)
        sg0_w, sg1_w, sg2_w = sg_0[win_idx], sg_1[win_idx], sg_2[win_idx]

        sscond = (
            (sg0_w < sg_0_th).all()
            and (np.abs(sg1_w) < sg_1_th).all()
            and (np.abs(sg2_w) < sg_2_th).all()
        )
        if sscond:
            return i, False

        bbcond = (sg0_w > (min_val_fwd + sg_0_th * 2)).all() and (sg1_w > sg_1_th).all()
        if bbcond:
            return min_idx_fwd, True

        end_idx = i
        current_window_size = _calc_window(window_size, sg0_w, sg1_w, sg_0_th, sg_1_th)

    return end_idx, False


def _resolve_collisions(
    events_raw, peak, start_idx, end_idx, b2b_start, b2b_end,
    sg_0, sg_0_th, simp_th, back_to_back,
):
    while events_raw:
        prev = events_raw[-1]
        if not (start_idx <= prev["end_idx"] or b2b_start or prev["b2b_end"]):
            break

        if not back_to_back:
            start_idx = min(prev["start_idx"], start_idx)
            end_idx = max(prev["end_idx"], end_idx)
            b2b_start = b2b_end = False
            if sg_0[prev["peak_idx"]] > sg_0[peak]:
                peak = prev["peak_idx"]
            events_raw.pop()
            continue

        p1, p2 = prev["peak_idx"], peak
        min_idx = int(np.argmin(sg_0[p1: p2 + 1])) + p1
        val_trough = sg_0[min_idx]
        h1 = max(1e-6, float(sg_0[p1]) - val_trough)
        h2 = max(1e-6, float(sg_0[p2]) - val_trough)

        if min(h1, h2) < simp_th * max(h1, h2):
            if h1 < h2:
                start_idx = min(prev["start_idx"], start_idx)
                b2b_start = prev["b2b_start"]
            else:
                start_idx = min(prev["start_idx"], start_idx)
                end_idx = max(prev["end_idx"], end_idx)
                b2b_start = prev["b2b_start"]
                peak = p1
            events_raw.pop()
            continue

        prev["end_idx"] = min_idx
        prev["b2b_end"] = True
        if start_idx < min_idx:
            start_idx = min_idx
            b2b_start = True
        break

    return events_raw, peak, start_idx, end_idx, b2b_start, b2b_end


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_events(
    rain_stormflow: pd.DataFrame,
    time_range: tuple[str, str] | None = None,
    *,
    back_to_back: bool = True,
    height_perc: float = 25.0,
    prom_perc: float = 25.0,
    width_perc: float = 10.0,
    distance: int = 48,
    window_length: int = 48,
    polyorder: int = 2,
    deriv_factor: float = 1.2,
    sg_0_per: float = 0.15,
    sg_1_per: float = 0.25,
    sg_2_per: float = 0.25,
    window_size: int = 72,
    simp_th: float = 0.5,
    verbose: bool = False,
) -> tuple[list[EventRecord], FilterResult]:
    """Detect storm events using Savitzky-Golay derivative segmentation.

    Parameters
    ----------
    rain_stormflow : pd.DataFrame
        Must contain ``datetime``, ``rain``, ``stormflow``.
    time_range : (start, end) str tuple, optional
        Restrict detection to this date range.
    back_to_back : bool
        When True, keep distinct events separated at troughs.
    height_perc, prom_perc, width_perc : float
        Percentile thresholds for peak detection (0–100 range).
    distance : int
        Minimum samples between peaks.
    window_length : int
        Base Savitzky-Golay window for sg_0.
    polyorder : int
        Savitzky-Golay polynomial order.
    deriv_factor : float
        Window multiplier for each successive derivative pass.
    sg_0_per, sg_1_per, sg_2_per : float
        Threshold multipliers for boundary detection.
    window_size : int
        Look-ahead/look-back window for boundary walks.
    simp_th : float
        Height-ratio threshold for merging near-equal adjacent peaks (0–1).
    verbose : bool

    Returns
    -------
    events : list[EventRecord]
        Detected events (event_id starts at 1).
    filter_result : FilterResult
        sg signals + computed thresholds for diagnostics.
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["stormflow"] = np.maximum(df["stormflow"].values, 0.0)

    if time_range is not None:
        s, e = pd.to_datetime(time_range[0]), pd.to_datetime(time_range[1])
        df = df[(df["datetime"] >= s) & (df["datetime"] <= e)].reset_index(drop=True)

    if len(df) < 5:
        fr = apply_savgol_filter(df, window_length, polyorder, deriv_factor)
        return [], compute_thresholds(fr, sg_0_per, sg_1_per, sg_2_per)

    n = len(df)
    safe_w = max(3 if 3 % 2 != 0 else 5, min(5, n if n % 2 != 0 else n - 1))
    sg_peak = np.maximum(
        savgol_filter(
            np.maximum(df["stormflow"].values, 0.0),
            window_length=safe_w,
            polyorder=min(polyorder, safe_w - 1),
            mode="interp",
        ),
        0.0,
    )
    peaks_idx = _find_peaks_with_thresholds(sg_peak, height_perc, prom_perc, width_perc, distance)

    if verbose:
        print(f"Candidate peaks: {len(peaks_idx)}")

    fr = apply_savgol_filter(df, window_length, polyorder, deriv_factor)
    fr = compute_thresholds(fr, sg_0_per, sg_1_per, sg_2_per)

    sg_0, sg_1, sg_2 = fr.sg_0, fr.sg_1, fr.sg_2
    sg_0_th = fr.thresholds["sg_0_th"]
    sg_1_th = fr.thresholds["sg_1_th"]
    sg_2_th = fr.thresholds["sg_2_th"]

    events_raw: list[dict] = []
    for idx, peak in enumerate(peaks_idx):
        start_idx, b2b_start = _backward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th)
        end_idx, b2b_end = _forward_walk(sg_0, sg_1, sg_2, peak, window_size, sg_0_th, sg_1_th, sg_2_th)

        if verbose:
            print(f"Peak {idx+1}/{len(peaks_idx)} idx={peak} → [{start_idx}, {end_idx}] b2b=({b2b_start},{b2b_end})")

        events_raw, peak, start_idx, end_idx, b2b_start, b2b_end = _resolve_collisions(
            events_raw, peak, start_idx, end_idx, b2b_start, b2b_end,
            sg_0, sg_0_th, simp_th, back_to_back,
        )

        if end_idx > start_idx:
            events_raw.append({"peak_idx": peak, "start_idx": start_idx, "end_idx": end_idx,
                                "b2b_start": b2b_start, "b2b_end": b2b_end})

    events: list[EventRecord] = []
    for i, raw in enumerate(events_raw):
        s_idx, e_idx = raw["start_idx"], raw["end_idx"]
        slice_df = df.iloc[s_idx: e_idx + 1]
        total_rain = float(slice_df["rain"].sum())
        total_flow = float(np.maximum(slice_df["stormflow"].values, 0.0).sum())
        ae = total_flow / total_rain if total_rain > 0 else 0.0
        events.append(
            EventRecord(
                event_id=i + 1,
                start_datetime=pd.Timestamp(slice_df["datetime"].iloc[0]),
                end_datetime=pd.Timestamp(slice_df["datetime"].iloc[-1]),
                peak_datetime=pd.Timestamp(df["datetime"].iloc[raw["peak_idx"]]),
                total_rain=total_rain,
                total_flow=total_flow,
                effective_area=ae,
                peak_flow=float(slice_df["stormflow"].max()),
                b2b_start=bool(raw["b2b_start"]),
                b2b_end=bool(raw["b2b_end"]),
            )
        )

    if verbose:
        print(f"Detected {len(events)} events.")

    return events, fr


__all__ = [
    "EventRecord",
    "detect_events",
    "events_to_dataframe",
    "load_events_from_csv",
]
