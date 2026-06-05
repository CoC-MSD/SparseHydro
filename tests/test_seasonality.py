"""Tests for SeasonalityModel — Fourier-series sanitary base-flow estimator."""

import unittest

import numpy as np
import pandas as pd

from sparsehydro.rdii.seasonality import SeasonalityModel


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(n: int = 24 * 7, freq: str = "h") -> pd.DataFrame:
    """Return a minimal DataFrame with a datetime column."""
    t = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({"datetime": t})


def _make_df_with_features(n: int = 48) -> pd.DataFrame:
    df = _make_df(n)
    dt = pd.to_datetime(df["datetime"])
    df["hour_of_day"] = dt.dt.hour + dt.dt.minute / 60.0
    df["day_of_week"] = dt.dt.dayofweek.astype(float)
    df["day_of_year"] = dt.dt.dayofyear.astype(float)
    return df


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestSeasonalityModelConstruction(unittest.TestCase):

    def test_default_periods(self):
        m = SeasonalityModel()
        self.assertIn("hour_of_day", m.periods)
        self.assertIn("day_of_week", m.periods)
        self.assertIn("day_of_year", m.periods)

    def test_custom_periods(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0})
        self.assertEqual(list(m.periods.keys()), ["hour_of_day"])

    def test_n_terms_stored(self):
        m = SeasonalityModel(n_terms=3)
        self.assertEqual(m.n_terms, 3)

    def test_output_name_stored(self):
        m = SeasonalityModel(output_name="q_san")
        self.assertEqual(m.output_name, "q_san")

    def test_invalid_n_terms_raises(self):
        with self.assertRaises(ValueError):
            SeasonalityModel(n_terms=0)

    def test_invalid_coeff_bounds_raises(self):
        with self.assertRaises(ValueError):
            SeasonalityModel(coeff_bounds=(10.0, -10.0))


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestSeasonalityModelInitialize(unittest.TestCase):

    def test_parameter_count_default(self):
        # 3 periods × 5 terms × 2 (a + b) = 30
        m = SeasonalityModel(n_terms=5)
        m.initialize()
        self.assertEqual(len(m.scalar_parameter_names), 30)

    def test_parameter_count_custom(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0, "day_of_year": 365.25}, n_terms=3)
        m.initialize()
        # 2 periods × 3 terms × 2 = 12
        self.assertEqual(len(m.scalar_parameter_names), 12)

    def test_parameter_names_format(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=2)
        m.initialize()
        names = m.scalar_parameter_names
        self.assertIn("a_hour_of_day_1", names)
        self.assertIn("b_hour_of_day_1", names)
        self.assertIn("a_hour_of_day_2", names)
        self.assertIn("b_hour_of_day_2", names)

    def test_default_coeff_values_zero(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=2)
        m.initialize()
        for name in m.scalar_parameter_names:
            self.assertEqual(m.get_scalar_parameter(name).value, 0.0)

    def test_coeff_bounds_applied(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1,
                             coeff_bounds=(-100.0, 100.0))
        m.initialize()
        p = m.get_scalar_parameter("a_hour_of_day_1")
        self.assertEqual(p.lower_bound, -100.0)
        self.assertEqual(p.upper_bound, 100.0)

    def test_state_initialized(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        self.assertTrue(m.is_initialized())


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------

class TestSeasonalityModelPrepare(unittest.TestCase):

    def test_auto_compute_hod(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        m.validate()
        df = _make_df(n=24)
        m.prepare(df)
        # hour_of_day should be auto-computed from datetime
        self.assertIn("hour_of_day", m._features)
        hod = m._features["hour_of_day"]
        self.assertEqual(len(hod), 24)

    def test_auto_compute_dow(self):
        m = SeasonalityModel(periods={"day_of_week": 7.0}, n_terms=1)
        m.initialize()
        m.validate()
        df = _make_df(n=7)
        m.prepare(df)
        self.assertIn("day_of_week", m._features)

    def test_auto_compute_doy(self):
        m = SeasonalityModel(periods={"day_of_year": 365.25}, n_terms=1)
        m.initialize()
        m.validate()
        df = _make_df(n=365)
        m.prepare(df)
        self.assertIn("day_of_year", m._features)

    def test_explicit_column_used_when_present(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        m.validate()
        df = _make_df(n=5)
        df["hour_of_day"] = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        m.prepare(df)
        np.testing.assert_array_equal(m._features["hour_of_day"], [1.0, 2.0, 3.0, 4.0, 5.0])

    def test_missing_non_standard_column_raises(self):
        m = SeasonalityModel(periods={"custom_feature": 12.0}, n_terms=1)
        m.initialize()
        m.validate()
        df = _make_df(n=5)  # no "custom_feature" column
        with self.assertRaises(ValueError):
            m.prepare(df)

    def test_state_prepared(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        m.validate()
        m.prepare(_make_df())
        self.assertTrue(m.is_prepared())


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestSeasonalityModelPredict(unittest.TestCase):

    def setUp(self):
        self.m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1,
                                  output_name="sanitary_cfs")
        self.m.initialize()
        self.m.validate()
        # 25 hourly steps: hod = 0, 1, ..., 24
        self.df = _make_df(n=25)
        self.m.prepare(self.df)

    def test_predict_returns_dataframe(self):
        out = self.m.predict()
        self.assertIsInstance(out, pd.DataFrame)

    def test_predict_columns(self):
        out = self.m.predict()
        self.assertIn("datetime", out.columns)
        self.assertIn("sanitary_cfs", out.columns)

    def test_all_zero_when_coeffs_zero(self):
        out = self.m.predict()
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), 0.0)

    def test_cosine_only(self):
        # a=1, b=0 → output = cos(2π·n·hod/24)
        self.m.get_scalar_parameter("a_hour_of_day_1").value = 1.0
        out = self.m.predict()
        hod = self.m._features["hour_of_day"]
        expected = np.cos(2.0 * np.pi * 1 * hod / 24.0)
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), expected, atol=1e-12)

    def test_sine_only(self):
        self.m.get_scalar_parameter("b_hour_of_day_1").value = 1.0
        out = self.m.predict()
        hod = self.m._features["hour_of_day"]
        expected = np.sin(2.0 * np.pi * 1 * hod / 24.0)
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), expected, atol=1e-12)

    def test_multiple_harmonics_superposition(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=2)
        m.initialize()
        m.validate()
        df = _make_df(n=24)
        m.prepare(df)
        m.get_scalar_parameter("a_hour_of_day_1").value = 1.0
        m.get_scalar_parameter("a_hour_of_day_2").value = 0.5
        out = m.predict()
        hod = m._features["hour_of_day"]
        expected = (np.cos(2 * np.pi * 1 * hod / 24)
                    + 0.5 * np.cos(2 * np.pi * 2 * hod / 24))
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), expected, atol=1e-12)

    def test_multi_period_additive(self):
        m = SeasonalityModel(
            periods={"hour_of_day": 24.0, "day_of_week": 7.0},
            n_terms=1,
        )
        m.initialize()
        m.validate()
        df = _make_df(n=24)
        m.prepare(df)
        m.get_scalar_parameter("a_hour_of_day_1").value = 1.0
        m.get_scalar_parameter("b_day_of_week_1").value = 2.0
        out = m.predict()
        hod = m._features["hour_of_day"]
        dow = m._features["day_of_week"]
        expected = (np.cos(2 * np.pi * hod / 24.0)
                    + 2.0 * np.sin(2 * np.pi * dow / 7.0))
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), expected, atol=1e-12)

    def test_state_predicted(self):
        self.m.predict()
        self.assertTrue(self.m.is_predicted())

    def test_output_length_matches_input(self):
        out = self.m.predict()
        self.assertEqual(len(out), 25)


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------

class TestSeasonalityModelFinalize(unittest.TestCase):

    def test_finalize_clears_state(self):
        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        m.validate()
        m.prepare(_make_df())
        m.predict()
        m.finalize()
        self.assertTrue(m.is_finalized())
        self.assertIsNone(m._datetime)
        self.assertEqual(m._features, {})


# ---------------------------------------------------------------------------
# CalibrationProblem integration
# ---------------------------------------------------------------------------

class TestSeasonalityCalibrationIntegration(unittest.TestCase):

    def test_calibration_problem_param_count(self):
        try:
            from sparsehydro.calibration import CalibrationProblem, MSE
        except ImportError:
            self.skipTest("calibration extras not installed")

        df = _make_df(n=48)
        observed = np.sin(np.arange(48, dtype=float) / 4)

        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=3)
        m.initialize()
        m.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=m,
            data=df,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df_: df_["sanitary_cfs"].to_numpy(),
            },
        )
        # 1 period × 3 terms × 2 (a+b) = 6
        self.assertEqual(problem.n_params, 6)

    def test_evaluate_runs(self):
        try:
            from sparsehydro.calibration import CalibrationProblem, MSE
        except ImportError:
            self.skipTest("calibration extras not installed")

        df = _make_df(n=24)
        observed = np.zeros(24)

        m = SeasonalityModel(periods={"hour_of_day": 24.0}, n_terms=1)
        m.initialize()
        m.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=m,
            data=df,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df_: df_["sanitary_cfs"].to_numpy(),
            },
        )
        x = np.array([0.0, 0.0])  # a=0, b=0 → output = 0 = observed → MSE = 0
        F = problem.evaluate(x)
        self.assertAlmostEqual(float(F[0]), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
