"""Dry-weather-flow (DWF) disaggregation by daily-pattern decomposition.

This module separates a sewer-flow (or level/velocity) time series into a slowly
varying *baseflow* component (sanitary diurnal pattern plus groundwater
infiltration, GWI) and a *stormflow* component (the rainfall-derived response).
It is a Python port of the MATLAB ``DailyPattern`` script used by MSDGC.

The algorithm builds a typical *weekly* diurnal pattern from the dry-weather
portions of the record, fits that pattern against the smoothed series to flag
storm-affected periods, corrects the pattern for slow GWI drift using the clean
windows that precede each storm, and finally subtracts the resulting baseflow
from the observed flow to obtain stormflow.

Pipeline
--------
1. Savitzky-Golay smoothing of the raw flow.
2. Generation of four candidate weekly patterns (differencing, median,
   weekday/weekend and single-day) at the native sub-daily resolution.
3. Selection of a pattern based on the length of the record.
4. Construction of a per-point base series from the selected weekly pattern.
5. Pattern-fit residual computation and outlier (storm) detection.
6. Baseflow correction from clean pre-storm windows, interpolated across time.
7. ``stormflow = flow - baseflow``.

The original MATLAB code is hard-wired for 5-minute data (288 samples/day,
2016 samples/week).  This port infers the sampling step from the timestamps and
generalises the index arithmetic to any regular timestep via ``points_per_day``.

Notes
-----
MATLAB's ``weekday`` returns ``1`` for Sunday through ``7`` for Saturday, and the
weekly index is built with Sunday as the first day.  pandas uses
``dayofweek == 0`` for Monday, so the Sunday-based day number is computed as
``(dayofweek + 1) % 7``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

__all__ = [
    "DryWeatherResult",
    "disaggregate_dry_weather",
    "generate_weekly_patterns",
]

# Pattern keys and their human-readable names (mirroring the MATLAB PATTERN struct).
PATTERN_NAMES: dict[str, str] = {
    "diff": "7-day by Differencing",
    "median": "7-day by Median",
    "weekday_weekend": "2-day Weekday/Weekend",
    "one_day": "1-day Median",
}

# A known Sunday at midnight, used to synthesise a reference week when rebuilding
# the weekday/weekend pattern onto an (unshifted) Sunday-first weekly grid.
_REFERENCE_SUNDAY = pd.Timestamp("2023-01-01")  # 2023-01-01 is a Sunday.


@dataclass
class DryWeatherResult:
    """Container for the output of :func:`disaggregate_dry_weather`.

    :ivar datetime: Timestamps of the (regularised) series.
    :vartype datetime: pandas.Series
    :ivar flow: Observed flow after missing-value handling.
    :vartype flow: numpy.ndarray
    :ivar smoothed: Savitzky-Golay smoothed flow used to build the pattern.
    :vartype smoothed: numpy.ndarray
    :ivar pattern: Per-point base series from the selected weekly pattern
        *before* GWI correction (the MATLAB ``RawBaseFlow``).
    :vartype pattern: numpy.ndarray
    :ivar baseflow: Corrected baseflow (pattern plus interpolated GWI
        correction).
    :vartype baseflow: numpy.ndarray
    :ivar stormflow: ``flow - baseflow``.
    :vartype stormflow: numpy.ndarray
    :ivar outliers: Boolean mask of storm-affected (outlier) points.
    :vartype outliers: numpy.ndarray
    :ivar pattern_name: Key of the selected weekly pattern (see
        :data:`PATTERN_NAMES`).
    :vartype pattern_name: str
    :ivar weekly_patterns: The four candidate weekly patterns as
        ``points_per_week``-length arrays.
    :vartype weekly_patterns: dict
    :ivar points_per_day: Number of samples per day inferred from the timestamps.
    :vartype points_per_day: int
    """

    datetime: pd.Series
    flow: np.ndarray
    smoothed: np.ndarray
    pattern: np.ndarray
    baseflow: np.ndarray
    stormflow: np.ndarray
    outliers: np.ndarray
    pattern_name: str
    weekly_patterns: dict[str, np.ndarray] = field(default_factory=dict)
    points_per_day: int = 0

    def to_dataframe(self) -> pd.DataFrame:
        """Return the disaggregation results as a tidy :class:`pandas.DataFrame`.

        :returns: Frame with columns ``datetime``, ``flow``, ``smoothed``,
            ``pattern``, ``baseflow``, ``stormflow``, ``outlier``.
        :rtype: pandas.DataFrame
        """
        return pd.DataFrame(
            {
                "datetime": np.asarray(self.datetime),
                "flow": self.flow,
                "smoothed": self.smoothed,
                "pattern": self.pattern,
                "baseflow": self.baseflow,
                "stormflow": self.stormflow,
                "outlier": self.outliers,
            }
        )


# ---------------------------------------------------------------------------
# Index helpers
# ---------------------------------------------------------------------------
def _dow_sunday(index: pd.DatetimeIndex) -> np.ndarray:
    """Day of week with Sunday=0 .. Saturday=6 (matches MATLAB ``weekday-1``).

    :param index: Timestamps.
    :type index: pandas.DatetimeIndex
    :returns: Sunday-based day-of-week number for each timestamp.
    :rtype: numpy.ndarray
    """
    return (index.dayofweek.to_numpy() + 1) % 7


def _within_day_index(
    index: pd.DatetimeIndex,
    points_per_day: int,
    step_seconds: float,
    shift_hours: float = 0.0,
) -> np.ndarray:
    """Index of each timestamp within its (optionally shifted) day, 0-based.

    :param index: Timestamps.
    :type index: pandas.DatetimeIndex
    :param points_per_day: Number of samples per day.
    :type points_per_day: int
    :param step_seconds: Sampling step in seconds.
    :type step_seconds: float
    :param shift_hours: Shift applied before computing the day boundary.  The
        MATLAB weekday/weekend pattern shifts by 18 hours so that the "day"
        starts at 6 PM.
    :type shift_hours: float
    :returns: 0-based within-day index for each timestamp.
    :rtype: numpy.ndarray
    """
    shifted = index - pd.Timedelta(hours=shift_hours)
    seconds = (shifted - shifted.normalize()).to_numpy() / np.timedelta64(1, "s")
    idx = np.floor(seconds / step_seconds).astype(int)
    return np.clip(idx, 0, points_per_day - 1)


def _week_index(
    index: pd.DatetimeIndex, points_per_day: int, step_seconds: float
) -> np.ndarray:
    """Sunday-first index of each timestamp within its week, 0-based.

    Equivalent to the MATLAB ``1 + 288*(weekday-1) + 12*Hr + floor(MN/5)`` for
    5-minute data (here generalised and 0-based).

    :param index: Timestamps.
    :type index: pandas.DatetimeIndex
    :param points_per_day: Number of samples per day.
    :type points_per_day: int
    :param step_seconds: Sampling step in seconds.
    :type step_seconds: float
    :returns: 0-based within-week index for each timestamp.
    :rtype: numpy.ndarray
    """
    points_per_week = 7 * points_per_day
    widx = _dow_sunday(index) * points_per_day + _within_day_index(
        index, points_per_day, step_seconds
    )
    return np.clip(widx, 0, points_per_week - 1)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def _is_outlier(values: np.ndarray) -> np.ndarray:
    """Flag outliers using MATLAB ``isoutlier`` default (median +/- 3 scaled MAD).

    An element is an outlier when it lies more than three scaled median absolute
    deviations (``1.4826 * MAD``) from the median.  NaNs are never flagged.

    :param values: Values to test.
    :type values: numpy.ndarray
    :returns: Boolean mask flagging outliers.
    :rtype: numpy.ndarray
    """
    values = np.asarray(values, dtype=float)
    mask = ~np.isnan(values)
    if mask.sum() == 0:
        return np.zeros_like(values, dtype=bool)
    med = np.median(values[mask])
    mad = np.median(np.abs(values[mask] - med))
    scaled_mad = 1.4826 * mad
    out = np.zeros_like(values, dtype=bool)
    out[mask] = np.abs(values[mask] - med) > 3 * scaled_mad
    return out


def _median_by_group(values: np.ndarray, group: np.ndarray, n_groups: int) -> np.ndarray:
    """Median of *values* grouped by integer *group* labels, reindexed 0..n-1.

    Empty groups yield NaN, which is then filled by periodic interpolation so the
    resulting pattern has no gaps.

    :param values: Values to aggregate.
    :type values: numpy.ndarray
    :param group: Integer group label for each value.
    :type group: numpy.ndarray
    :param n_groups: Total number of groups (output length).
    :type n_groups: int
    :returns: Per-group medians reindexed to ``0..n_groups-1`` with gaps filled.
    :rtype: numpy.ndarray
    """
    series = pd.Series(values).groupby(group).median()
    out = np.full(n_groups, np.nan)
    out[series.index.to_numpy()] = series.to_numpy()
    return _fill_pattern_nans(out)


def _fill_pattern_nans(pattern: np.ndarray) -> np.ndarray:
    """Fill NaNs in a (periodic) weekly pattern by interpolation.

    :param pattern: Weekly pattern that may contain NaNs.
    :type pattern: numpy.ndarray
    :returns: Pattern with NaNs filled by interpolation and median fallback.
    :rtype: numpy.ndarray
    """
    s = pd.Series(pattern)
    if s.notna().sum() == 0:
        return np.zeros_like(pattern)
    s = s.interpolate(limit_direction="both")
    s = s.fillna(float(np.nanmedian(pattern)))
    return s.to_numpy()


# ---------------------------------------------------------------------------
# Weekly pattern generators
# ---------------------------------------------------------------------------
def generate_weekly_patterns(
    smoothed: np.ndarray,
    index: pd.DatetimeIndex,
    points_per_day: int,
    step_seconds: float,
) -> dict[str, np.ndarray]:
    """Generate the four candidate weekly patterns from a smoothed series.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param index: Timestamps aligned with *smoothed*.
    :type index: pandas.DatetimeIndex
    :param points_per_day: Samples per day.
    :type points_per_day: int
    :param step_seconds: Sampling step in seconds.
    :type step_seconds: float
    :returns: Mapping of pattern key (``"diff"``, ``"median"``,
        ``"weekday_weekend"``, ``"one_day"``) to a ``7 * points_per_day``-length
        weekly pattern array.
    :rtype: dict
    """
    smoothed = np.asarray(smoothed, dtype=float)
    ppd = points_per_day
    ppw = 7 * ppd
    week_idx = _week_index(index, ppd, step_seconds)
    day_idx = _within_day_index(index, ppd, step_seconds)

    patterns = {
        "median": _pattern_median(smoothed, week_idx, ppw),
        "one_day": _pattern_one_day(smoothed, day_idx, ppd),
        "diff": _pattern_diff(smoothed, week_idx, ppw),
        "weekday_weekend": _pattern_weekday_weekend(
            smoothed, index, ppd, step_seconds
        ),
    }
    return patterns


def _pattern_median(smoothed: np.ndarray, week_idx: np.ndarray, ppw: int) -> np.ndarray:
    """Type 2: median of the smoothed series at each weekly index.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param week_idx: Within-week index for each sample.
    :type week_idx: numpy.ndarray
    :param ppw: Points per week (output length).
    :type ppw: int
    :returns: Weekly median pattern.
    :rtype: numpy.ndarray
    """
    return _median_by_group(smoothed, week_idx, ppw)


def _pattern_one_day(smoothed: np.ndarray, day_idx: np.ndarray, ppd: int) -> np.ndarray:
    """Type 4: single-day median pattern tiled across seven days.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param day_idx: Within-day index for each sample.
    :type day_idx: numpy.ndarray
    :param ppd: Points per day.
    :type ppd: int
    :returns: Single-day median profile tiled across the week.
    :rtype: numpy.ndarray
    """
    day_profile = _median_by_group(smoothed, day_idx, ppd)
    return np.tile(day_profile, 7)


def _pattern_diff(smoothed: np.ndarray, week_idx: np.ndarray, ppw: int) -> np.ndarray:
    """Type 1: weekly pattern reconstructed from median first-differences.

    The median difference at each weekly index is accumulated and then
    de-trended so that the pattern is periodic over the week.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param week_idx: Within-week index for each sample.
    :type week_idx: numpy.ndarray
    :param ppw: Points per week (output length).
    :type ppw: int
    :returns: De-trended weekly pattern.
    :rtype: numpy.ndarray
    """
    diffs = np.diff(smoothed)
    diff_idx = week_idx[:-1]
    median_diff = _median_by_group(diffs, diff_idx, ppw)

    cumulative = np.zeros(ppw)
    first_vals = smoothed[week_idx == 0]
    cumulative[0] = np.median(first_vals) if first_vals.size else 0.0
    for k in range(1, ppw):
        cumulative[k] = cumulative[k - 1] + median_diff[k - 1]

    # Remove residual trend so the pattern closes on itself over the week.
    scale = (cumulative[0] - (cumulative[ppw - 1] + median_diff[ppw - 1])) / ppw
    return cumulative + np.arange(ppw) * scale


def _pattern_weekday_weekend(
    smoothed: np.ndarray,
    index: pd.DatetimeIndex,
    ppd: int,
    step_seconds: float,
) -> np.ndarray:
    """Type 3: separate weekday and weekend day-profiles.

    The week is shifted by 18 hours so each "day" starts at 6 PM, and the
    weekend is defined as 6 PM Friday through 6 PM Sunday (the shifted
    Sunday-based day numbers 5 and 6).  Weekday and weekend day-profiles are
    computed over the shifted day index, then mapped back onto an unshifted
    Sunday-first weekly grid via a synthetic reference week.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param index: Timestamps aligned with *smoothed*.
    :type index: pandas.DatetimeIndex
    :param ppd: Points per day.
    :type ppd: int
    :param step_seconds: Sampling step in seconds.
    :type step_seconds: float
    :returns: Weekly pattern combining weekday and weekend profiles.
    :rtype: numpy.ndarray
    """
    ppw = 7 * ppd
    shift = 18.0

    shifted_day_idx = _within_day_index(index, ppd, step_seconds, shift_hours=shift)
    shifted_dow = _dow_sunday(index - pd.Timedelta(hours=shift))
    is_weekend = np.isin(shifted_dow, (5, 6))

    weekday_profile = _median_by_group(
        smoothed[~is_weekend], shifted_day_idx[~is_weekend], ppd
    )
    weekend_profile = _median_by_group(
        smoothed[is_weekend], shifted_day_idx[is_weekend], ppd
    )

    # Reference Sunday-first week at the native resolution.
    step = pd.Timedelta(seconds=step_seconds)
    ref = pd.DatetimeIndex(_REFERENCE_SUNDAY + np.arange(ppw) * step)
    ref_day_idx = _within_day_index(ref, ppd, step_seconds, shift_hours=shift)
    ref_weekend = np.isin(_dow_sunday(ref - pd.Timedelta(hours=shift)), (5, 6))

    as7day = np.where(
        ref_weekend, weekend_profile[ref_day_idx], weekday_profile[ref_day_idx]
    )
    return as7day


# ---------------------------------------------------------------------------
# Pattern fit and storm detection
# ---------------------------------------------------------------------------
def _pattern_fit(smoothed: np.ndarray, base: np.ndarray, window: int) -> np.ndarray:
    """Per-point pattern-fit residual (MATLAB ``DPatternFit``).

    For each point the smoothed series and the base pattern are normalised by
    their mean over a trailing window and compared over forward, centred and
    backward windows; the minimum of the three RMS-style residuals is returned.

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param base: Per-point base pattern series.
    :type base: numpy.ndarray
    :param window: Trailing/look-around window length in samples.
    :type window: int
    :returns: The minimum fit residual at each point.
    :rtype: numpy.ndarray
    """
    n = smoothed.shape[0]
    fit = np.full(n, np.nan)
    eps = 1e-12

    for j in range(n):
        imin = max(0, j - window)
        imax = min(j + window - 1, n - 1)

        ts_norm = np.mean(smoothed[imin : j + 1])
        dp_norm = np.mean(base[imin : j + 1])
        if abs(ts_norm) < eps or abs(dp_norm) < eps:
            continue

        def _resid(lo: int, hi: int) -> float:
            ts_seg = smoothed[lo : hi + 1] / ts_norm
            dp_seg = base[lo : hi + 1] / dp_norm
            return float(np.sqrt(np.sum((ts_seg - dp_seg) ** 2)))

        forward = _resid(j, imax)
        centered = _resid(imin, imax)
        backward = _resid(imin, j)
        fit[j] = min(forward, centered, backward)

    return fit


def _outlier_rules(
    fit: np.ndarray,
    after: int,
    before: int,
    fill: int,
) -> np.ndarray:
    """Expand and fill outlier flags (MATLAB ``OutlierRules``).

    :param fit: Per-point fit residual.
    :type fit: numpy.ndarray
    :param after: Number of points *after* a raw outlier to also flag.
    :type after: int
    :param before: Number of points *before* a raw outlier to also flag.
    :type before: int
    :param fill: Window used to fill short gaps between outliers.
    :type fill: int
    :returns: Boolean outlier mask after expansion and gap-filling.
    :rtype: numpy.ndarray
    """
    n = fit.shape[0]
    raw = _is_outlier(fit)
    outlier = np.zeros(n, dtype=bool)

    # Expand each raw outlier by [-after, +before].
    if after == 0 and before == 0:
        outlier = raw.copy()
    else:
        for j in range(after, n - before):
            if raw[j - after : j + before + 1].any():
                outlier[j] = True

    if fill <= 0:
        return outlier

    # Fill gaps that sit between two outlier clusters within ``fill`` points.
    snapshot = outlier.copy()
    for j in range(1, n - 1):
        imin = max(0, j - fill)
        imax = min(j + fill - 1, n - 1)
        if (
            not outlier[j - 1]
            and not outlier[j]
            and snapshot[imin : max(imin, j - 1)].any()
            and snapshot[j + 1 : imax + 1].any()
        ):
            outlier[imin : imax + 1] = True

    if outlier[:fill].any():
        outlier[:fill] = True
    return outlier


def _ave_before_storm(
    smoothed: np.ndarray,
    base: np.ndarray,
    outlier: np.ndarray,
    times: np.ndarray,
    window: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Baseflow corrections from clean windows preceding each storm.

    For each point, if the immediately following point is an outlier and the
    trailing window is outlier-free, the mean difference between the smoothed
    series and the base pattern over that window is recorded as a candidate
    correction.  Candidates are filtered by their GWI slope and by an outlier
    test on the correction magnitude (MATLAB ``f_avebeforestorm``).

    :param smoothed: Smoothed flow series.
    :type smoothed: numpy.ndarray
    :param base: Per-point base pattern series.
    :type base: numpy.ndarray
    :param outlier: Boolean storm/outlier mask.
    :type outlier: numpy.ndarray
    :param times: Per-point timestamps as floats (nanoseconds since epoch).
    :type times: numpy.ndarray
    :param window: Trailing clean-window length in samples.
    :type window: int
    :returns: Tuple ``(anchor_index, corrections)`` where *anchor_index* holds
        the point indices of the retained correction anchors and *corrections*
        holds the baseflow correction at each retained anchor.
    :rtype: tuple[numpy.ndarray, numpy.ndarray]
    """
    n = smoothed.shape[0]
    idxs: list[int] = []
    corr: list[float] = []
    base_means: list[float] = []
    anchor_times: list[float] = []

    for j in range(window, n - 1):
        imin = max(0, j - window)
        imax = min(j + window - 1, n - 1)
        if imax + 1 >= n:
            continue
        if outlier[imax + 1] and not outlier[imin : imax + 1].any():
            ts_mean = float(np.mean(smoothed[imin : imax + 1]))
            cdp_mean = float(np.mean(base[imin : imax + 1]))
            idxs.append(j)
            corr.append(ts_mean - cdp_mean)
            base_means.append(cdp_mean)
            anchor_times.append(float(times[j]))

    if not idxs:
        return np.array([], dtype=int), np.array([], dtype=float)

    idxs_a = np.array(idxs, dtype=int)
    corr_a = np.array(corr, dtype=float)
    base_means_a = np.array(base_means, dtype=float)
    times_a = np.array(anchor_times, dtype=float)

    # GWI slope between consecutive anchors (first slope is zero).
    gwi_slope = np.zeros_like(corr_a)
    dt = np.diff(times_a)
    dt[dt == 0] = np.nan
    gwi_slope[1:] = np.diff(base_means_a) / dt
    keep_slope = np.abs(gwi_slope) < np.mean(np.abs(gwi_slope)) if gwi_slope.size else None
    if keep_slope is not None and keep_slope.any():
        idxs_a = idxs_a[keep_slope]
        corr_a = corr_a[keep_slope]

    # Drop anchors whose correction is itself an outlier.
    if corr_a.size:
        good = ~_is_outlier(corr_a)
        if good.any():
            idxs_a = idxs_a[good]
            corr_a = corr_a[good]

    return idxs_a, corr_a


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------
def _safe_odd_window(window_length: int, n: int) -> int:
    """Clamp *window_length* to an odd value no larger than *n*.

    :param window_length: Desired window length.
    :type window_length: int
    :param n: Number of available samples (upper bound).
    :type n: int
    :returns: Odd window length in ``[3, n]``.
    :rtype: int
    """
    w = min(window_length, n if n % 2 == 1 else n - 1)
    if w % 2 == 0:
        w -= 1
    return max(w, 3)


def _select_pattern_key(span_days: float) -> str:
    """Choose a weekly pattern from the record length (MATLAB ``SPat`` logic).

    :param span_days: Length of the record in days.
    :type span_days: float
    :returns: Pattern key (``"one_day"``, ``"weekday_weekend"`` or
        ``"median"``).
    :rtype: str
    """
    if span_days < 22:
        return "one_day"
    if span_days < 36:
        return "weekday_weekend"
    return "median"


def disaggregate_dry_weather(
    df: pd.DataFrame,
    datetime_col: str = "datetime",
    flow_col: str = "flow",
    *,
    pattern: str = "auto",
    smooth_window: int = 48,
    smooth_polyorder: int = 2,
    missing_value: float | None = -999.0,
    fill_missing_with_baseflow: bool = True,
) -> DryWeatherResult:
    """Disaggregate a flow time series into baseflow and stormflow.

    :param df: Input data containing a datetime column and a flow column.
    :type df: pandas.DataFrame
    :param datetime_col: Name of the datetime column.
    :type datetime_col: str
    :param flow_col: Name of the flow column.
    :type flow_col: str
    :param pattern: Weekly pattern to use: ``"auto"`` (select by record length,
        the default), or one of ``"diff"``, ``"median"``, ``"weekday_weekend"``,
        ``"one_day"``.
    :type pattern: str
    :param smooth_window: Savitzky-Golay window length applied to the raw flow
        before pattern generation.
    :type smooth_window: int
    :param smooth_polyorder: Savitzky-Golay polynomial order.
    :type smooth_polyorder: int
    :param missing_value: Sentinel value treated as missing (replaced with
        NaN).  Set to ``None`` to disable.
    :type missing_value: float or None
    :param fill_missing_with_baseflow: If ``True``, missing/zero flow points are
        filled with the computed baseflow before stormflow is calculated.
    :type fill_missing_with_baseflow: bool
    :returns: The disaggregation results.
    :rtype: DryWeatherResult
    :raises ValueError: If the series has fewer than two points or an irregular
        timestep cannot be inferred.
    """
    if pattern not in ("auto", *PATTERN_NAMES):
        raise ValueError(
            f"pattern must be 'auto' or one of {tuple(PATTERN_NAMES)}, got {pattern!r}"
        )

    data = df[[datetime_col, flow_col]].copy()
    data[datetime_col] = pd.to_datetime(data[datetime_col])
    data = data.sort_values(datetime_col).drop_duplicates(datetime_col)
    index = pd.DatetimeIndex(data[datetime_col].to_numpy())
    flow = data[flow_col].to_numpy(dtype=float).copy()

    if flow.shape[0] < 2:
        raise ValueError("At least two samples are required for disaggregation.")

    if missing_value is not None:
        flow[flow == missing_value] = np.nan
    missing_mask = np.isnan(flow) | (flow == 0.0)

    # Infer the regular sampling step (resolution-agnostic).
    deltas = np.diff(index.values) / np.timedelta64(1, "s")
    step_seconds = float(np.median(deltas))
    if step_seconds <= 0:
        raise ValueError("Could not infer a positive sampling step from timestamps.")
    points_per_day = int(round(86400.0 / step_seconds))
    if points_per_day < 1:
        raise ValueError("Sampling step is coarser than one day; not supported.")

    # Smoothing requires a gap-free series; interpolate missing for smoothing only.
    flow_for_smooth = pd.Series(flow).interpolate(limit_direction="both").to_numpy()
    flow_for_smooth = np.nan_to_num(flow_for_smooth, nan=0.0)
    window = _safe_odd_window(smooth_window, flow_for_smooth.shape[0])
    smoothed = savgol_filter(
        flow_for_smooth, window_length=window, polyorder=smooth_polyorder, mode="interp"
    )

    # Candidate weekly patterns.
    weekly_patterns = generate_weekly_patterns(
        smoothed, index, points_per_day, step_seconds
    )

    span_days = (index[-1] - index[0]) / np.timedelta64(1, "D")
    pattern_key = _select_pattern_key(span_days) if pattern == "auto" else pattern
    as7day = weekly_patterns[pattern_key]

    # Map the weekly pattern onto every timestamp (MATLAB F_baseTScreate).
    week_idx = _week_index(index, points_per_day, step_seconds)
    base = as7day[week_idx]

    # Storm detection via pattern-fit residual outliers.
    fit = _pattern_fit(smoothed, base, window=points_per_day)
    outliers = _outlier_rules(
        fit,
        after=points_per_day,
        before=max(points_per_day // 6, 1),
        fill=max(points_per_day // 2, 1),
    )

    # Baseflow correction from clean pre-storm windows.
    times = index.asi8.astype(float)
    anchor_idx, corrections = _ave_before_storm(
        smoothed, base, outliers, times, window=max(points_per_day // 2, 1)
    )

    if anchor_idx.size:
        xp = np.concatenate(([times[0]], times[anchor_idx], [times[-1]]))
        fp = np.concatenate(([corrections[0]], corrections, [corrections[-1]]))
        # np.interp needs strictly increasing xp; collapse duplicate endpoints.
        order = np.argsort(xp, kind="stable")
        xp, fp = xp[order], fp[order]
        uniq = np.concatenate(([True], np.diff(xp) > 0))
        correction = np.interp(times, xp[uniq], fp[uniq])
    else:
        correction = np.zeros_like(base)

    baseflow = base + correction

    flow_out = flow.copy()
    if fill_missing_with_baseflow:
        flow_out[missing_mask] = baseflow[missing_mask]
    else:
        flow_out = np.nan_to_num(flow_out, nan=0.0)

    stormflow = flow_out - baseflow

    return DryWeatherResult(
        datetime=pd.Series(index, name=datetime_col),
        flow=flow_out,
        smoothed=smoothed,
        pattern=base,
        baseflow=baseflow,
        stormflow=stormflow,
        outliers=outliers,
        pattern_name=pattern_key,
        weekly_patterns=weekly_patterns,
        points_per_day=points_per_day,
    )
