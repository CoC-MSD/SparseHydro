"""Tests for the dry-weather-flow disaggregation module.

A synthetic flow series with a known diurnal pattern, a slow groundwater drift
and an injected storm is used to exercise the disaggregation pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from sparsehydro.filters import (
    DryWeatherResult,
    disaggregate_dry_weather,
    generate_weekly_patterns,
)
from sparsehydro.filters.dryweather import (
    _dow_sunday,
    _is_outlier,
    _select_pattern_key,
    _week_index,
)


# 5-minute sampling, matching the original MATLAB resolution.
STEP_SECONDS = 300.0
PPD = 288


def _make_series(
    n_days: int,
    *,
    storm: bool = False,
    drift: float = 0.0,
    start: str = "2023-01-01",
) -> pd.DataFrame:
    """Build a synthetic 5-minute diurnal flow series.

    Parameters
    ----------
    n_days : int
        Number of days of data.
    storm : bool
        If True, inject a triangular storm hydrograph on a middle day.
    drift : float
        Linear baseflow drift added across the record (units/day).
    """
    periods = n_days * PPD
    index = pd.date_range(start=start, periods=periods, freq="5min")
    hours = index.hour.to_numpy() + index.minute.to_numpy() / 60.0
    # Smooth diurnal pattern: low at night, peaks mid-morning and evening.
    diurnal = (
        2.0
        + 0.8 * np.sin((hours - 6) / 24 * 2 * np.pi)
        + 0.4 * np.sin((hours - 6) / 24 * 4 * np.pi)
    )
    drift_series = drift * (np.arange(periods) / PPD)
    flow = diurnal + drift_series

    if storm:
        # Triangular storm on day n_days // 2, lasting ~12 hours.
        storm_start = (n_days // 2) * PPD + 6 * 12  # 6 AM
        storm_len = 12 * 12  # 12 hours
        peak = storm_start + storm_len // 2
        for k in range(storm_start, storm_start + storm_len):
            ramp = 1.0 - abs(k - peak) / (storm_len / 2)
            flow[k] += 5.0 * max(ramp, 0.0)

    return pd.DataFrame({"datetime": index, "flow": flow})


def test_dow_sunday_convention() -> None:
    # 2023-01-01 is a Sunday -> 0; 2023-01-07 is Saturday -> 6.
    index = pd.DatetimeIndex(["2023-01-01", "2023-01-02", "2023-01-07"])
    assert list(_dow_sunday(index)) == [0, 1, 6]


def test_week_index_range_and_start() -> None:
    index = pd.date_range("2023-01-01", periods=PPD * 7, freq="5min")
    widx = _week_index(index, PPD, STEP_SECONDS)
    assert widx.min() == 0
    assert widx.max() == PPD * 7 - 1
    # Sunday midnight is index 0.
    assert widx[0] == 0


def test_is_outlier_flags_spike() -> None:
    values = np.ones(100)
    values[50] = 100.0
    out = _is_outlier(values)
    assert out[50]
    assert out.sum() == 1


def test_select_pattern_key_by_span() -> None:
    assert _select_pattern_key(10) == "one_day"
    assert _select_pattern_key(30) == "weekday_weekend"
    assert _select_pattern_key(60) == "median"


def test_generate_weekly_patterns_shapes() -> None:
    df = _make_series(14)
    index = pd.DatetimeIndex(df["datetime"].to_numpy())
    smoothed = df["flow"].to_numpy()
    patterns = generate_weekly_patterns(smoothed, index, PPD, STEP_SECONDS)
    assert set(patterns) == {"diff", "median", "weekday_weekend", "one_day"}
    for arr in patterns.values():
        assert arr.shape == (PPD * 7,)
        assert np.all(np.isfinite(arr))


def test_one_day_pattern_is_tiled() -> None:
    df = _make_series(14)
    index = pd.DatetimeIndex(df["datetime"].to_numpy())
    patterns = generate_weekly_patterns(df["flow"].to_numpy(), index, PPD, STEP_SECONDS)
    one_day = patterns["one_day"]
    # All seven days share the same single-day profile.
    for d in range(1, 7):
        np.testing.assert_allclose(one_day[:PPD], one_day[d * PPD : (d + 1) * PPD])


def test_disaggregate_returns_result() -> None:
    df = _make_series(40)
    result = disaggregate_dry_weather(df)
    assert isinstance(result, DryWeatherResult)
    assert result.points_per_day == PPD
    assert result.pattern_name == "median"  # 40-day span -> median pattern
    n = len(df)
    for arr in (result.flow, result.baseflow, result.stormflow, result.outliers):
        assert arr.shape == (n,)


def test_dry_series_baseflow_tracks_flow() -> None:
    # With no storm, baseflow should closely follow the flow and stormflow ~ 0.
    df = _make_series(40)
    result = disaggregate_dry_weather(df)
    assert np.mean(np.abs(result.stormflow)) < 0.2
    np.testing.assert_allclose(
        result.baseflow, result.flow, atol=0.6
    )


def test_storm_detected_and_stormflow_positive() -> None:
    df = _make_series(40, storm=True)
    result = disaggregate_dry_weather(df)
    assert result.outliers.any()
    # The peak stormflow should be clearly positive during the storm.
    assert result.stormflow.max() > 2.0


def test_flow_reconstructs_from_components() -> None:
    df = _make_series(40, storm=True)
    result = disaggregate_dry_weather(df)
    np.testing.assert_allclose(
        result.flow, result.baseflow + result.stormflow, atol=1e-9
    )


def test_to_dataframe_columns() -> None:
    df = _make_series(40)
    result = disaggregate_dry_weather(df)
    out = result.to_dataframe()
    assert list(out.columns) == [
        "datetime",
        "flow",
        "smoothed",
        "pattern",
        "baseflow",
        "stormflow",
        "outlier",
    ]
    assert len(out) == len(df)


def test_invalid_pattern_raises() -> None:
    df = _make_series(40)
    with pytest.raises(ValueError):
        disaggregate_dry_weather(df, pattern="bogus")


def test_too_few_samples_raises() -> None:
    df = pd.DataFrame({"datetime": ["2023-01-01"], "flow": [1.0]})
    with pytest.raises(ValueError):
        disaggregate_dry_weather(df)


def test_missing_values_filled_with_baseflow() -> None:
    df = _make_series(40)
    df.loc[100:120, "flow"] = -999.0
    result = disaggregate_dry_weather(df)
    # Filled points should equal the baseflow there (stormflow ~ 0).
    filled = result.stormflow[100:121]
    assert np.allclose(filled, 0.0, atol=1e-9)


def test_custom_pattern_selection() -> None:
    df = _make_series(40)
    result = disaggregate_dry_weather(df, pattern="one_day")
    assert result.pattern_name == "one_day"
