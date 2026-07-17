"""Tests for SeasonalityModel — discrete peaking-factor seasonal flow estimator.

The new SeasonalityModel uses discrete peaking factors for hour-of-day (24 bins),
day-of-week (7 bins), and month-of-year (12 bins), rather than the old Fourier series.

Formula: output[t] = baseline * sum_d( w_d * pf_d_norm[category_d(t)] )
where pf_d_norm = pf_d / mean(pf_d), and sum(w_d) == 1.
"""

import unittest

import numpy as np
import pandas as pd

from sparsehydro.models import SeasonalityModel


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_df(n: int = 24 * 7, freq: str = "h") -> pd.DataFrame:
    """Return a minimal DataFrame with a datetime column."""
    t = pd.date_range("2024-01-01", periods=n, freq=freq)
    return pd.DataFrame({"datetime": t})


# ---------------------------------------------------------------------------
# Construction & validation
# ---------------------------------------------------------------------------

class TestSeasonalityModelConstruction(unittest.TestCase):

    def test_default_all_dims_active(self):
        m = SeasonalityModel()
        self.assertTrue(m.include_hour)
        self.assertTrue(m.include_dow)
        self.assertTrue(m.include_month)

    def test_include_hour_only(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        self.assertTrue(m.include_hour)
        self.assertFalse(m.include_dow)
        self.assertFalse(m.include_month)

    def test_output_name_stored(self):
        m = SeasonalityModel(output_name="q_san")
        self.assertEqual(m.output_name, "q_san")

    def test_default_output_name_is_flow(self):
        m = SeasonalityModel()
        self.assertEqual(m.output_name, "flow")

    def test_no_dims_raises(self):
        with self.assertRaises(ValueError):
            SeasonalityModel(include_hour=False, include_dow=False, include_month=False)


# ---------------------------------------------------------------------------
# initialize()
# ---------------------------------------------------------------------------

class TestSeasonalityModelInitialize(unittest.TestCase):

    def test_parameter_count_all_dims(self):
        # 1 baseline + 3 weights + 24 pf_hour + 7 pf_dow + 12 pf_month
        # But these are vector parameters — scalar_parameter_names contains only scalar params
        # Scalar params: baseline, w_hour, w_dow, w_month = 4
        m = SeasonalityModel(include_hour=True, include_dow=True, include_month=True)
        m.initialize()
        # scalar: baseline + 3 dimension weights = 4
        self.assertEqual(len(m.scalar_parameter_names), 4)

    def test_parameter_count_hour_only(self):
        # scalar: baseline + w_hour = 2
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        self.assertEqual(len(m.scalar_parameter_names), 2)

    def test_parameter_count_hour_and_month(self):
        # scalar: baseline + w_hour + w_month = 3
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=True)
        m.initialize()
        self.assertEqual(len(m.scalar_parameter_names), 3)

    def test_baseline_registered(self):
        m = SeasonalityModel()
        m.initialize()
        self.assertIn("baseline", m.scalar_parameter_names)

    def test_weight_params_registered_all_dims(self):
        m = SeasonalityModel()
        m.initialize()
        self.assertIn("w_hour", m.scalar_parameter_names)
        self.assertIn("w_dow", m.scalar_parameter_names)
        self.assertIn("w_month", m.scalar_parameter_names)

    def test_weight_params_only_for_active_dims(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        self.assertIn("w_hour", m.scalar_parameter_names)
        self.assertNotIn("w_dow", m.scalar_parameter_names)
        self.assertNotIn("w_month", m.scalar_parameter_names)

    def test_pf_hour_vector_param_registered(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        pf = m.get_vector_parameter("pf_hour")
        self.assertEqual(len(pf.values), 24)

    def test_pf_dow_vector_param_registered(self):
        m = SeasonalityModel(include_hour=False, include_dow=True, include_month=False)
        m.initialize()
        pf = m.get_vector_parameter("pf_dow")
        self.assertEqual(len(pf.values), 7)

    def test_pf_month_vector_param_registered(self):
        m = SeasonalityModel(include_hour=False, include_dow=False, include_month=True)
        m.initialize()
        pf = m.get_vector_parameter("pf_month")
        self.assertEqual(len(pf.values), 12)

    def test_pf_hour_default_values_are_ones(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        np.testing.assert_allclose(m.get_vector_parameter("pf_hour").values, 1.0)

    def test_default_weight_equals_one_over_n_dims(self):
        m = SeasonalityModel(include_hour=True, include_dow=True, include_month=False)
        m.initialize()
        self.assertAlmostEqual(m.get_scalar_parameter("w_hour").value, 0.5)
        self.assertAlmostEqual(m.get_scalar_parameter("w_dow").value, 0.5)

    def test_sum_w_constraints_registered(self):
        m = SeasonalityModel()
        m.initialize()
        self.assertIn("sum_w_leq_1", m.inequality_constraint_names)
        self.assertIn("sum_w_geq_1", m.inequality_constraint_names)

    def test_state_initialized(self):
        m = SeasonalityModel()
        m.initialize()
        self.assertTrue(m.is_initialized())


# ---------------------------------------------------------------------------
# validate()
# ---------------------------------------------------------------------------

class TestSeasonalityModelValidate(unittest.TestCase):

    def test_validate_passes_with_defaults(self):
        m = SeasonalityModel()
        m.initialize()
        self.assertTrue(m.validate())
        self.assertTrue(m.is_validated())


# ---------------------------------------------------------------------------
# prepare()
# ---------------------------------------------------------------------------

class TestSeasonalityModelPrepare(unittest.TestCase):

    def test_prepare_sets_state(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()
        m.prepare(_make_df(n=24))
        self.assertTrue(m.is_prepared())

    def test_prepare_missing_datetime_raises(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()
        df = pd.DataFrame({"other": [1, 2, 3]})
        with self.assertRaises((KeyError, ValueError)):
            m.prepare(df)

    def test_prepare_caches_hour_index(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()
        df = _make_df(n=24)
        m.prepare(df)
        self.assertIsNotNone(m._h)
        self.assertEqual(len(m._h), 24)

    def test_prepare_caches_dow_index(self):
        m = SeasonalityModel(include_hour=False, include_dow=True, include_month=False)
        m.initialize()
        m.validate()
        df = _make_df(n=7)
        m.prepare(df)
        self.assertIsNotNone(m._dow)

    def test_prepare_caches_month_index(self):
        m = SeasonalityModel(include_hour=False, include_dow=False, include_month=True)
        m.initialize()
        m.validate()
        df = _make_df(n=365)
        m.prepare(df)
        self.assertIsNotNone(m._month)
        # month is 0-indexed: 0=Jan, 11=Dec
        self.assertTrue(np.all(m._month >= 0))
        self.assertTrue(np.all(m._month <= 11))


# ---------------------------------------------------------------------------
# predict()
# ---------------------------------------------------------------------------

class TestSeasonalityModelPredict(unittest.TestCase):

    def setUp(self):
        self.m = SeasonalityModel(
            include_hour=True, include_dow=False, include_month=False,
            output_name="sanitary_cfs",
        )
        self.m.initialize()
        self.m.validate()
        self.df = _make_df(n=25)
        self.m.prepare(self.df)

    def test_predict_returns_dataframe(self):
        out = self.m.predict()
        self.assertIsInstance(out, pd.DataFrame)

    def test_predict_columns(self):
        out = self.m.predict()
        self.assertIn("datetime", out.columns)
        self.assertIn("sanitary_cfs", out.columns)

    def test_predict_output_length_matches_input(self):
        out = self.m.predict()
        self.assertEqual(len(out), 25)

    def test_uniform_pf_yields_baseline(self):
        """When all peaking factors are equal (default = 1.0), output == baseline."""
        self.m.get_scalar_parameter("baseline").value = 5.0
        out = self.m.predict()
        # pf_norm = 1.0/1.0 = 1.0 everywhere; w_hour = 1.0; output = 5.0 * 1.0 * 1.0 = 5.0
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), 5.0, atol=1e-12)

    def test_baseline_zero_gives_zero_output(self):
        self.m.get_scalar_parameter("baseline").value = 0.0
        out = self.m.predict()
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), 0.0, atol=1e-12)

    def test_predict_state_predicted(self):
        self.m.predict()
        self.assertTrue(self.m.is_predicted())

    def test_peaking_factor_doubles_for_specific_hour(self):
        """Setting one hour's pf to 2x while keeping others at 1 doubles that hour's output."""
        self.m.get_scalar_parameter("baseline").value = 1.0
        pf = self.m.get_vector_parameter("pf_hour").values.copy()
        pf[0] = 2.0  # hour 0 gets double peaking factor
        self.m.get_vector_parameter("pf_hour").values = pf
        out = self.m.predict()

        # pf_norm = pf / mean(pf); mean = (2 + 23*1)/24 = 25/24
        # pf_norm[0] = 2 / (25/24) = 48/25 = 1.92
        # pf_norm[others] = 1 / (25/24) = 24/25 = 0.96
        mean_pf = pf.mean()
        h_array = self.m._h
        expected_norm = pf[h_array] / mean_pf
        expected_out = 1.0 * expected_norm
        np.testing.assert_allclose(out["sanitary_cfs"].to_numpy(), expected_out, atol=1e-12)

    def test_multi_dim_weighted_sum(self):
        """Multi-dim model computes weighted sum of normalized peaking factors."""
        m = SeasonalityModel(
            include_hour=True, include_dow=True, include_month=False,
            output_name="flow",
        )
        m.initialize()
        m.validate()
        df = _make_df(n=48)
        m.prepare(df)
        m.get_scalar_parameter("baseline").value = 2.0
        m.get_scalar_parameter("w_hour").value = 0.7
        m.get_scalar_parameter("w_dow").value = 0.3
        out = m.predict()

        # Compute expected manually
        w_sum = 1.0  # 0.7 + 0.3
        pf_h = m.get_vector_parameter("pf_hour").values
        pf_d = m.get_vector_parameter("pf_dow").values
        pf_h_norm = pf_h / pf_h.mean()
        pf_d_norm = pf_d / pf_d.mean()
        w_h = 0.7 / w_sum
        w_d = 0.3 / w_sum
        expected = 2.0 * (w_h * pf_h_norm[m._h] + w_d * pf_d_norm[m._dow])
        np.testing.assert_allclose(out["flow"].to_numpy(), expected, atol=1e-12)


# ---------------------------------------------------------------------------
# inequality_constraints()
# ---------------------------------------------------------------------------

class TestSeasonalityModelConstraints(unittest.TestCase):

    def test_constraints_return_two_values(self):
        m = SeasonalityModel()
        m.initialize()
        g = m.inequality_constraints()
        self.assertEqual(len(g), 2)

    def test_default_weights_sum_to_one(self):
        """Default weights sum to 1 — both constraints should equal 0."""
        m = SeasonalityModel()
        m.initialize()
        g = m.inequality_constraints()
        # sum(w) - 1 == 0 and 1 - sum(w) == 0
        self.assertAlmostEqual(g[0], 0.0)
        self.assertAlmostEqual(g[1], 0.0)

    def test_overweight_violates_first_constraint(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.get_scalar_parameter("w_hour").value = 1.5  # sum > 1
        g = m.inequality_constraints()
        self.assertGreater(g[0], 0.0)  # infeasible

    def test_underweight_violates_second_constraint(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.get_scalar_parameter("w_hour").value = 0.5  # sum < 1
        g = m.inequality_constraints()
        self.assertGreater(g[1], 0.0)  # infeasible


# ---------------------------------------------------------------------------
# finalize()
# ---------------------------------------------------------------------------

class TestSeasonalityModelFinalize(unittest.TestCase):

    def test_finalize_clears_state(self):
        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()
        m.prepare(_make_df())
        m.predict()
        m.finalize()
        self.assertTrue(m.is_finalized())
        self.assertIsNone(m._datetime)
        self.assertIsNone(m._h)
        self.assertIsNone(m._dow)
        self.assertIsNone(m._month)


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
        observed = np.ones(48)

        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=m,
            data=df,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df_: df_["flow"].to_numpy(),
            },
        )
        # scalar params: baseline + w_hour = 2 (calibratable)
        # vector params (pf_hour 24) are calibratable too — total depends on impl
        # Just check it's >= 2
        self.assertGreaterEqual(problem.n_params, 2)

    def test_evaluate_uniform_gives_baseline(self):
        try:
            from sparsehydro.calibration import CalibrationProblem, MSE
        except ImportError:
            self.skipTest("calibration extras not installed")

        df = _make_df(n=24)
        # Target = 1.0 everywhere (default baseline=1.0, uniform pf → output = 1.0)
        observed = np.ones(24)

        m = SeasonalityModel(include_hour=True, include_dow=False, include_month=False)
        m.initialize()
        m.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=m,
            data=df,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df_: df_["flow"].to_numpy(),
            },
        )
        # Calibratable scalar params only: baseline=1.0, w_hour=1.0 (2 params)
        x = np.array([1.0] * problem.n_params)
        F = problem.evaluate(x)
        self.assertAlmostEqual(float(F[0]), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
