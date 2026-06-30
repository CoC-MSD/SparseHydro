"""Shared data utilities for MSDGC flow-monitoring datasets.

This module collects helpers that operate on the time-series data produced by
the service clients in :mod:`data.services`:

* :func:`read_tsf` parses FlowFinity ``.tsf`` time-series files into pandas
  DataFrames.
* :func:`add_time_features` augments a datetime-indexed (or datetime-columned)
  DataFrame with decimal time-of-week / time-of-day / time-of-year features
  commonly used for seasonality modelling.

Example
-------
::

    from data.utilities import read_tsf, add_time_features

    df = read_tsf("downloads/lstStormTs.tsf")
    df = add_time_features(df, hour_of_day=True, day_of_week=True)
"""

from __future__ import annotations

import os
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd

__all__ = ["read_tsf", "add_time_features"]


def read_tsf(
    source: str | bytes | os.PathLike[str],
    *,
    start: Any = None,
    end: Any = None,
) -> pd.DataFrame:
    """Parse a FlowFinity ``.tsf`` time-series file into a DataFrame.

    The ``.tsf`` layout is::

        IDs:\t<title>
        Date/Time\t<Variable>
        <date-format>\t<unit>
        <datetime>\t<value>
        ...

    The first (``IDs:``) and third (units) lines are skipped; the second
    line provides column names.  The ``Date/Time`` column is parsed to
    datetimes and used as the index.

    :param source: A filesystem path, the raw file ``bytes``, or the decoded
        text content of a ``.tsf`` file.
    :param start: Optional inclusive lower datetime bound.
    :param end: Optional inclusive upper datetime bound.
    :returns: DataFrame indexed by ``Date/Time`` with the value column(s).
    :raises ValueError: If the content does not contain the expected header.
    """
    if isinstance(source, (str, os.PathLike)) and not (
        isinstance(source, str) and "\n" in source
    ):
        text = Path(source).read_text(encoding="utf-8-sig")
    elif isinstance(source, bytes):
        text = source.decode("utf-8-sig")
    else:
        text = str(source)

    lines = text.splitlines()
    if len(lines) < 3:
        raise ValueError("Not a valid .tsf file: fewer than 3 header lines.")

    # Line 0 = "IDs:\t<title>", line 1 = headers, line 2 = units, data follows.
    header = lines[1].split("\t")
    data_text = "\n".join(lines[3:])

    frame = pd.read_csv(
        StringIO(data_text),
        sep="\t",
        names=header,
        engine="python",
    )

    time_col = header[0]
    frame[time_col] = pd.to_datetime(frame[time_col], errors="coerce")
    frame = frame.dropna(subset=[time_col]).set_index(time_col).sort_index()

    if start is not None:
        frame = frame[frame.index >= pd.to_datetime(start)]
    if end is not None:
        frame = frame[frame.index <= pd.to_datetime(end)]

    return frame


def add_time_features(
    df: pd.DataFrame,
    *,
    hour_of_day: bool = False,
    day_of_week: bool = False,
    day_of_year: bool = False,
    datetime_col: str | None = None,
) -> pd.DataFrame:
    """Add decimal time-feature columns to a DataFrame.

    Each feature is opt-in via its boolean flag.  Timestamps are taken from
    *datetime_col* when given, otherwise from the DataFrame's
    :class:`~pandas.DatetimeIndex`, otherwise from a column named
    ``"datetime"``.

    The decimal features are:

    * ``hour_of_day`` — hour of the day in ``[0, 24)``, computed as
      ``hour + minute / 60 + second / 3600``.
    * ``day_of_week`` — day of the week in ``[0, 7)`` with Monday as ``0``,
      computed as ``dayofweek + hour_of_day / 24``.
    * ``day_of_year`` — day of the year (``1``-based) plus the fractional day,
      computed as ``dayofyear + hour_of_day / 24``.

    :param df: Source DataFrame.  It is not modified; a copy is returned.
    :param hour_of_day: When ``True``, add a ``"hour_of_day"`` column.
    :param day_of_week: When ``True``, add a ``"day_of_week"`` column.
    :param day_of_year: When ``True``, add a ``"day_of_year"`` column.
    :param datetime_col: Name of the column holding timestamps.  When ``None``,
        the DatetimeIndex is used if present, else a ``"datetime"`` column.
    :returns: A copy of *df* with the requested feature column(s) appended.
    :raises KeyError: If *datetime_col* is given but absent, or if no datetime
        source can be located.
    :raises TypeError: If the located source cannot be parsed as datetimes.
    """
    result = df.copy()

    if datetime_col is not None:
        if datetime_col not in result.columns:
            raise KeyError(f"Column {datetime_col!r} not found in DataFrame.")
        dt = pd.to_datetime(result[datetime_col])
    elif isinstance(result.index, pd.DatetimeIndex):
        dt = result.index.to_series()
    elif "datetime" in result.columns:
        dt = pd.to_datetime(result["datetime"])
    else:
        raise KeyError(
            "No datetime source found: pass datetime_col=, provide a "
            "DatetimeIndex, or include a 'datetime' column."
        )

    accessor = dt.dt
    decimal_hour = accessor.hour + accessor.minute / 60.0 + accessor.second / 3600.0
    decimal_hour = pd.Series(decimal_hour.to_numpy(), index=result.index)

    if hour_of_day:
        result["hour_of_day"] = decimal_hour
    if day_of_week:
        dow = pd.Series(accessor.dayofweek.to_numpy(), index=result.index)
        result["day_of_week"] = dow + decimal_hour / 24.0
    if day_of_year:
        doy = pd.Series(accessor.dayofyear.to_numpy(), index=result.index)
        result["day_of_year"] = doy + decimal_hour / 24.0

    return result
