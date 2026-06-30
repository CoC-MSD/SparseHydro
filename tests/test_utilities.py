"""Tests for data.utilities — unittest.TestCase."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from data.utilities import add_time_features, read_tsf


_STORM_TSF = (
    "IDs:\tTC-NB-001 Flow Edited Revised\n"
    "Date/Time\tStormFlow\n"
    "M/d/yyyy\tMGD\n"
    "1/25/2022 12:40:00 PM\t0.10\n"
    "1/25/2022 12:45:00 PM\t0.20\n"
    "1/26/2022 12:40:00 PM\t0.30\n"
    "1/27/2022 12:40:00 PM\t0.40\n"
)


# ===========================================================================
# read_tsf
# ===========================================================================

class TestReadTsf(unittest.TestCase):

    def test_read_from_text(self):
        df = read_tsf(_STORM_TSF)
        self.assertEqual(len(df), 4)
        self.assertEqual(df.index.name, "Date/Time")
        self.assertEqual(list(df.columns), ["StormFlow"])

    def test_read_from_bytes(self):
        df = read_tsf(_STORM_TSF.encode("utf-8"))
        self.assertEqual(len(df), 4)

    def test_read_from_path(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / "storm.tsf"
            p.write_text(_STORM_TSF, encoding="utf-8")
            df = read_tsf(p)
            self.assertEqual(len(df), 4)

    def test_date_filtering(self):
        df = read_tsf(_STORM_TSF, start="2022-01-26")
        self.assertEqual(len(df), 2)

    def test_sorted_index(self):
        df = read_tsf(_STORM_TSF)
        self.assertTrue(df.index.is_monotonic_increasing)

    def test_too_few_lines_raises(self):
        with self.assertRaises(ValueError):
            read_tsf("IDs:\ttitle\nDate/Time\tX")


# ===========================================================================
# add_time_features
# ===========================================================================

class TestAddTimeFeatures(unittest.TestCase):

    def _indexed_frame(self) -> pd.DataFrame:
        # 2022-01-03 is a Monday (dayofweek == 0).
        idx = pd.DatetimeIndex(
            [
                "2022-01-03 00:00:00",  # Monday midnight
                "2022-01-03 06:00:00",  # Monday 06:00
                "2022-01-04 12:30:00",  # Tuesday 12:30
            ],
            name="Date/Time",
        )
        return pd.DataFrame({"value": [1.0, 2.0, 3.0]}, index=idx)

    def test_no_flags_leaves_columns_unchanged(self):
        df = self._indexed_frame()
        out = add_time_features(df)
        self.assertEqual(list(out.columns), ["value"])

    def test_does_not_mutate_input(self):
        df = self._indexed_frame()
        add_time_features(df, hour_of_day=True)
        self.assertEqual(list(df.columns), ["value"])

    def test_hour_of_day_from_index(self):
        out = add_time_features(self._indexed_frame(), hour_of_day=True)
        self.assertIn("hour_of_day", out.columns)
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 0.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[1], 6.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[2], 12.5)

    def test_day_of_week_from_index(self):
        out = add_time_features(self._indexed_frame(), day_of_week=True)
        self.assertAlmostEqual(out["day_of_week"].iloc[0], 0.0)
        self.assertAlmostEqual(out["day_of_week"].iloc[1], 6.0 / 24.0)
        self.assertAlmostEqual(out["day_of_week"].iloc[2], 1.0 + 12.5 / 24.0)

    def test_day_of_year_from_index(self):
        out = add_time_features(self._indexed_frame(), day_of_year=True)
        self.assertAlmostEqual(out["day_of_year"].iloc[0], 3.0)
        self.assertAlmostEqual(out["day_of_year"].iloc[2], 4.0 + 12.5 / 24.0)

    def test_all_flags_together(self):
        out = add_time_features(
            self._indexed_frame(),
            hour_of_day=True,
            day_of_week=True,
            day_of_year=True,
        )
        for col in ("hour_of_day", "day_of_week", "day_of_year"):
            self.assertIn(col, out.columns)

    def test_named_datetime_column(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(
                    ["2022-01-03 06:00:00", "2022-01-04 12:30:00"]
                ),
                "value": [1.0, 2.0],
            }
        )
        out = add_time_features(df, hour_of_day=True, datetime_col="datetime")
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 6.0)
        self.assertAlmostEqual(out["hour_of_day"].iloc[1], 12.5)

    def test_default_datetime_column(self):
        df = pd.DataFrame(
            {
                "datetime": pd.to_datetime(["2022-01-03 06:00:00"]),
                "value": [1.0],
            }
        )
        out = add_time_features(df, hour_of_day=True)
        self.assertAlmostEqual(out["hour_of_day"].iloc[0], 6.0)

    def test_missing_named_column_raises(self):
        df = self._indexed_frame()
        with self.assertRaises(KeyError):
            add_time_features(df, hour_of_day=True, datetime_col="nope")

    def test_no_datetime_source_raises(self):
        df = pd.DataFrame({"value": [1.0, 2.0]})
        with self.assertRaises(KeyError):
            add_time_features(df, hour_of_day=True)


if __name__ == "__main__":
    unittest.main()
