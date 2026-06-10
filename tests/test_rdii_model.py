"""Tests for the sparsehydro RDII module using unittest.TestCase.

Covers:
  - IAModel (recovery, depletion, excess computation)
  - RTKTriangle and triangular_uh (unit-area, shape)
  - Objective functions (peak_weighted_mse, nash_sutcliffe)
  - RDIIModel full lifecycle and physical constraints
  - RDIIOptimizer (returns CalibrationResult; conditional on pymoo)
  - Visualization functions (conditional on plotly)
"""

from __future__ import annotations

import math
import unittest

import numpy as np
import pandas as pd

from sparsehydro.rdii import (
    IAModel,
    RDIIModel,
    RTKTriangle,
    nash_sutcliffe,
    peak_weighted_mse,
    triangular_uh,
)

# ---------------------------------------------------------------------------
# Optional-dependency flags
# ---------------------------------------------------------------------------

try:
    import pymoo  # noqa: F401
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False

try:
    import plotly  # noqa: F401
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _make_sample_df(n: int = 48) -> pd.DataFrame:
    """Return an ``n``-hour DataFrame with a short storm event."""
    dates = pd.date_range("2024-01-01", periods=n, freq="h")
    rain = np.zeros(n)
    rain[5:10] = 5.0          # 5 consecutive hours of 5 mm
    temp = np.full(n, 15.0)   # below T_ref=20, above T_freeze=0
    return pd.DataFrame({
        "datetime": dates,
        "rainfall_mm": rain,
        "temperature_c": temp,
        "flow_cfs": np.random.default_rng(0).uniform(0, 10, n),
    })


def _prepared_model(n_triangles: int = 3) -> RDIIModel:
    """Return a fully prepared RDIIModel using the sample DataFrame (metric)."""
    m = RDIIModel(n_triangles=n_triangles, units="metric")
    m.initialize()
    m.validate()
    m.prepare(_make_sample_df())
    return m


# ---------------------------------------------------------------------------
# TestIAModel
# ---------------------------------------------------------------------------

class TestIAModel(unittest.TestCase):

    def test_default_state_is_ia_max(self):
        m = IAModel(ia_max=10.0)
        self.assertAlmostEqual(m.ia_avail, 10.0)

    def test_recovery_rate_above_freeze(self):
        m = IAModel(k0=0.05, kT=0.02, theta=0.1, T_ref=20.0, T_freeze=0.0)
        self.assertGreater(m.recovery_rate(temperature=15.0), 0.0)

    def test_recovery_rate_below_freeze(self):
        m = IAModel(T_freeze=5.0)
        self.assertAlmostEqual(m.recovery_rate(temperature=2.0), 0.0)

    def test_recovery_rate_at_T_ref(self):
        m = IAModel(k0=0.05, kT=0.02, theta=0.1, T_ref=20.0, T_freeze=0.0)
        # At T=T_ref: k = k0 + kT * exp(0) = k0 + kT
        self.assertAlmostEqual(m.recovery_rate(temperature=20.0), 0.05 + 0.02)

    def test_recovery_rate_none_uses_T_ref(self):
        m = IAModel(k0=0.05, kT=0.02, theta=0.1, T_ref=20.0, T_freeze=0.0)
        self.assertAlmostEqual(m.recovery_rate(None), m.recovery_rate(20.0))

    def test_step_dry_converges_to_ia_max(self):
        m = IAModel(ia_max=10.0, k0=1.0, kT=0.0)
        m.ia_avail = 0.0
        m.step_dry(dt_hours=100.0, temperature=20.0)
        self.assertAlmostEqual(m.ia_avail, 10.0, delta=1e-3)

    def test_step_dry_no_change_below_freeze(self):
        m = IAModel(ia_max=10.0, k0=0.1, T_freeze=5.0)
        m.ia_avail = 2.0
        m.step_dry(dt_hours=10.0, temperature=3.0)
        self.assertAlmostEqual(m.ia_avail, 2.0)

    def test_step_wet_depletes(self):
        m = IAModel(ia_max=10.0, k_dep=0.5)
        m.ia_avail = 10.0
        new_val = m.step_wet(delta_precip_mm=5.0)
        self.assertLess(new_val, 10.0)
        self.assertAlmostEqual(new_val, 10.0 * math.exp(-0.5 * 5.0))

    def test_step_wet_zero_no_change(self):
        m = IAModel(ia_max=10.0)
        m.ia_avail = 7.0
        m.step_wet(0.0)
        self.assertAlmostEqual(m.ia_avail, 7.0)

    def test_ia_avail_never_exceeds_ia_max(self):
        m = IAModel(ia_max=5.0, k0=10.0)
        m.ia_avail = 0.0
        for _ in range(10):
            m.step_dry(dt_hours=1.0, temperature=25.0)
        self.assertLessEqual(m.ia_avail, m.ia_max + 1e-12)

    def test_ia_avail_never_below_zero(self):
        m = IAModel(ia_max=5.0, k_dep=2.0)
        m.ia_avail = 1.0
        for _ in range(5):
            m.step_wet(100.0)
        self.assertGreaterEqual(m.ia_avail, 0.0)

    def test_compute_excess_zero_below_capacity(self):
        m = IAModel(ia_max=10.0)
        m.ia_avail = 10.0
        excess = m.compute_excess(rainfall_mm=2.0)
        self.assertAlmostEqual(excess, 0.0)

    def test_compute_excess_positive_when_saturated(self):
        m = IAModel(ia_max=1.0, k_dep=10.0)
        m.ia_avail = 0.1
        excess = m.compute_excess(rainfall_mm=5.0)
        self.assertGreater(excess, 0.0)

    def test_reset_restores_ia_max(self):
        m = IAModel(ia_max=8.0)
        m.ia_avail = 2.0
        m.reset()
        self.assertAlmostEqual(m.ia_avail, 8.0)

    def test_compute_excess_series_length(self):
        rain = np.array([0.0, 5.0, 10.0, 5.0, 0.0])
        result = IAModel.compute_excess_series(
            rainfall_mm=rain,
            dt_hours=np.ones(5),
            temperature=np.full(5, 20.0),
            ia_max=5.0, k0=0.05, kT=0.02, theta=0.1,
            T_ref=20.0, k_dep=0.3, T_freeze=0.0,
        )
        self.assertEqual(result.shape, (5,))
        self.assertTrue(np.all(result >= 0.0))

    def test_compute_excess_series_no_rain_is_zero(self):
        result = IAModel.compute_excess_series(
            rainfall_mm=np.zeros(10),
            dt_hours=np.ones(10),
            temperature=None,
            ia_max=5.0, k0=0.05, kT=0.0, theta=0.0,
            T_ref=20.0, k_dep=0.3, T_freeze=0.0,
        )
        np.testing.assert_allclose(result, 0.0)

    def test_compute_excess_series_freeze_suppresses_recovery(self):
        """Cold conditions → less inter-storm recovery → more excess in 2nd storm."""
        rain = np.array([10.0, 0.0, 0.0, 0.0, 10.0])
        kwargs = dict(
            rainfall_mm=rain, dt_hours=np.ones(5),
            ia_max=5.0, k0=0.1, kT=0.0, theta=0.0,
            T_ref=20.0, k_dep=1.0, T_freeze=0.0,
        )
        result_cold = IAModel.compute_excess_series(
            temperature=np.full(5, -5.0), **kwargs
        )
        result_warm = IAModel.compute_excess_series(
            temperature=np.full(5, 25.0), **kwargs
        )
        self.assertGreaterEqual(result_cold[4], result_warm[4])


# ---------------------------------------------------------------------------
# TestIAModelDesignDepletion
# ---------------------------------------------------------------------------

class TestIAModelDesignDepletion(unittest.TestCase):
    """Design contract: IA_consumed = ia_avail * (1 - exp(-k_dep * P)) is
    abstracted from the rainfall AND drained from the storage (mass-conserving),
    so antecedent wetness directly controls the response."""

    def test_mass_conservation(self):
        """Capacity drained equals the water abstracted from the rainfall."""
        m = IAModel(ia_max=10.0, k_dep=0.1)
        m.ia_avail = 10.0
        P = 5.0
        excess = m.compute_excess(rainfall_mm=P)
        consumed = 10.0 * (1.0 - math.exp(-0.1 * P))
        self.assertAlmostEqual(excess, P - consumed)
        self.assertAlmostEqual(10.0 - m.ia_avail, consumed)
        self.assertAlmostEqual(10.0 - m.ia_avail, P - excess)

    def test_k_dep_zero_means_no_abstraction(self):
        """k_dep = 0 → IA_consumed = 0 → excess = P (no degenerate threshold)."""
        rain = np.array([0.0, 2.0, 5.0])
        result = IAModel.compute_excess_series(
            rainfall_mm=rain, dt_hours=np.ones(3), temperature=None,
            ia_max=10.0, k0=0.05, kT=0.0, theta=0.0,
            T_ref=20.0, k_dep=0.0, T_freeze=0.0,
        )
        np.testing.assert_allclose(result, rain)

    def test_large_k_dep_approaches_threshold(self):
        """k_dep → ∞ → IA_consumed → ia_avail → excess → max(0, P - ia_avail)."""
        m = IAModel(ia_max=2.0, k_dep=1e6)
        m.ia_avail = 2.0
        excess = m.compute_excess(rainfall_mm=5.0)
        self.assertAlmostEqual(excess, 3.0, places=6)
        self.assertAlmostEqual(m.ia_avail, 0.0, places=6)

    def test_antecedent_contrast_same_storm(self):
        """A full bucket abstracts more from the same storm than a drained one."""
        full = IAModel(ia_max=2.0, k_dep=1.0)
        full.ia_avail = 2.0
        low = IAModel(ia_max=2.0, k_dep=1.0)
        low.ia_avail = 0.2
        self.assertLess(full.compute_excess(1.5), low.compute_excess(1.5))

    def test_dry_spell_then_back_to_back_storms(self):
        """After 30 dry days the first storm is mostly absorbed; the second,
        on the drained bucket, mostly runs off."""
        rain = np.zeros(32)
        rain[30] = 1.5
        rain[31] = 1.5
        result = IAModel.compute_excess_series(
            rainfall_mm=rain, dt_hours=np.full(32, 24.0), temperature=None,
            ia_max=2.0, k0=0.05, kT=0.0, theta=0.0,
            T_ref=20.0, k_dep=1.0, T_freeze=0.0,
        )
        # Day 30: consumed = 2*(1-e^-1.5) = 1.554 > 1.5 → excess 0
        # Day 31: bucket ~0.446 → consumed 0.347 → excess ~1.153
        self.assertAlmostEqual(result[30], 0.0)
        self.assertGreater(result[31], 1.0)

    def test_recovery_refills_between_storms(self):
        """Dry-interval recovery restores absorption for the next storm."""
        rain = np.array([5.0, 0.0, 0.0, 0.0, 1.0])
        kwargs = dict(
            rainfall_mm=rain, dt_hours=np.full(5, 24.0), temperature=None,
            ia_max=2.0, kT=0.0, theta=0.0,
            T_ref=20.0, k_dep=2.0, T_freeze=0.0,
        )
        fast = IAModel.compute_excess_series(k0=1.0, **kwargs)
        slow = IAModel.compute_excess_series(k0=0.0001, **kwargs)
        self.assertLess(fast[4], slow[4])


# ---------------------------------------------------------------------------
# TestRTKTriangle
# ---------------------------------------------------------------------------

class TestRTKTriangle(unittest.TestCase):

    def test_K_below_one_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RTKTriangle(R=0.05, T=1.0, K=0.5)
        self.assertIn("K must be > 1", str(ctx.exception))

    def test_K_equal_one_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RTKTriangle(R=0.05, T=1.0, K=1.0)
        self.assertIn("K must be > 1", str(ctx.exception))

    def test_T_zero_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RTKTriangle(R=0.05, T=0.0, K=2.0)
        self.assertIn("T must be > 0", str(ctx.exception))

    def test_R_negative_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RTKTriangle(R=-0.1, T=1.0, K=2.0)
        self.assertIn("R must be in", str(ctx.exception))

    def test_R_above_one_raises(self):
        with self.assertRaises(ValueError) as ctx:
            RTKTriangle(R=1.1, T=1.0, K=2.0)
        self.assertIn("R must be in", str(ctx.exception))

    def test_base_duration(self):
        tri = RTKTriangle(R=0.05, T=2.0, K=3.0)
        self.assertAlmostEqual(tri.base_duration, 8.0)  # 2*(1+3)

    def test_peak_ordinate(self):
        tri = RTKTriangle(R=0.05, T=2.0, K=3.0)
        # 2 / (2*(1+3)) = 0.25
        self.assertAlmostEqual(tri.peak_ordinate, 0.25)


# ---------------------------------------------------------------------------
# TestTriangularUH
# ---------------------------------------------------------------------------

class TestTriangularUH(unittest.TestCase):

    def test_unit_area_parametrized(self):
        """Unit-area property for several (T, K, dt) combinations."""
        cases = [
            (2.0, 2.0, 1.0),
            (3.0, 1.5, 0.5),
            (6.0, 3.0, 0.25),
            (1.0, 4.0, 0.1),
            (12.0, 2.0, 1.0),
        ]
        for T, K, dt in cases:
            with self.subTest(T=T, K=K, dt=dt):
                tri = RTKTriangle(R=0.1, T=T, K=K)
                ords = triangular_uh(tri, dt_hours=dt)
                np.testing.assert_allclose(
                    np.sum(ords) * dt, 1.0, rtol=1e-4,
                    err_msg=f"Unit area failed for T={T}, K={K}, dt={dt}",
                )

    def test_non_negative_ordinates(self):
        tri = RTKTriangle(R=0.1, T=3.0, K=1.5)
        ords = triangular_uh(tri, dt_hours=1.0)
        self.assertTrue(np.all(ords >= 0.0))

    def test_length_covers_base_duration(self):
        tri = RTKTriangle(R=0.05, T=2.0, K=2.0)
        dt = 1.0
        ords = triangular_uh(tri, dt_hours=dt)
        self.assertGreaterEqual(len(ords) * dt, tri.base_duration)

    def test_peak_step_near_T(self):
        tri = RTKTriangle(R=0.1, T=4.0, K=2.0)
        dt = 1.0
        ords = triangular_uh(tri, dt_hours=dt)
        peak_idx = int(np.argmax(ords))
        self.assertAlmostEqual(peak_idx, int(tri.T / dt), delta=1)

    def test_zero_dt_raises(self):
        tri = RTKTriangle(R=0.1, T=2.0, K=2.0)
        with self.assertRaises(ValueError) as ctx:
            triangular_uh(tri, dt_hours=0.0)
        self.assertIn("dt_hours must be > 0", str(ctx.exception))

    def test_custom_n_steps(self):
        tri = RTKTriangle(R=0.1, T=2.0, K=2.0)
        ords = triangular_uh(tri, dt_hours=1.0, n_steps=20)
        self.assertEqual(len(ords), 20)


# ---------------------------------------------------------------------------
# TestObjectives
# ---------------------------------------------------------------------------

class TestObjectives(unittest.TestCase):

    def test_pwmse_perfect_predictor(self):
        obs = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(peak_weighted_mse(obs, obs), 0.0)

    def test_pwmse_positive_for_imperfect(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([1.1, 1.9, 3.2])
        self.assertGreater(peak_weighted_mse(obs, pred), 0.0)

    def test_pwmse_shape_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            peak_weighted_mse(np.ones(3), np.ones(4))
        self.assertIn("Shape mismatch", str(ctx.exception))

    def test_pwmse_all_zero_obs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            peak_weighted_mse(np.zeros(3), np.ones(3))
        self.assertIn("all zeros", str(ctx.exception))

    def test_nse_perfect_predictor(self):
        obs = np.array([1.0, 2.0, 3.0, 2.0])
        self.assertAlmostEqual(nash_sutcliffe(obs, obs), 1.0)

    def test_nse_mean_predictor(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.full(3, np.mean(obs))
        self.assertAlmostEqual(nash_sutcliffe(obs, pred), 0.0)

    def test_nse_negative_for_poor_predictor(self):
        obs = np.array([1.0, 2.0, 3.0])
        pred = np.array([3.0, 2.0, 1.0])
        self.assertLess(nash_sutcliffe(obs, pred), 0.0)

    def test_nse_constant_obs_raises(self):
        with self.assertRaises(ValueError) as ctx:
            nash_sutcliffe(np.full(4, 2.0), np.array([1.0, 2.0, 3.0, 4.0]))
        self.assertIn("SS_tot is zero", str(ctx.exception))

    def test_nse_shape_mismatch_raises(self):
        with self.assertRaises(ValueError) as ctx:
            nash_sutcliffe(np.ones(3), np.ones(4))
        self.assertIn("Shape mismatch", str(ctx.exception))


# ---------------------------------------------------------------------------
# TestRDIIModel
# ---------------------------------------------------------------------------

class TestRDIIModel(unittest.TestCase):

    def setUp(self):
        self.sample_df = _make_sample_df()

    def _prepared(self, n_triangles: int = 3) -> RDIIModel:
        m = RDIIModel(n_triangles=n_triangles, units="metric")
        m.initialize()
        self.assertTrue(m.validate())
        m.prepare(self.sample_df)
        return m

    # --- Identity / registration ---

    def test_model_name(self):
        self.assertEqual(RDIIModel.model_name, "rdii")

    def test_registered_in_global_registry(self):
        from sparsehydro.registry import registry
        self.assertTrue(registry.is_registered("rdii"))

    # --- Construction ---

    def test_initial_state_created(self):
        self.assertTrue(RDIIModel().is_created())

    def test_zero_triangles_raises(self):
        with self.assertRaises(ValueError):
            RDIIModel(n_triangles=0)

    # --- initialize ---

    def test_initialize_param_count(self):
        for n in (1, 2, 3, 5):
            with self.subTest(n_triangles=n):
                m = RDIIModel(n_triangles=n)
                m.initialize()
                self.assertEqual(len(m.scalar_parameter_names), 8 + 3 * n)

    def test_initialize_ia_params_present(self):
        m = RDIIModel()
        m.initialize()
        ia_expected = {
            "ia_max", "ia_k0", "ia_kT", "ia_theta",
            "ia_T_ref", "ia_k_dep", "ia_T_freeze",
        }
        self.assertTrue(ia_expected.issubset(set(m.scalar_parameter_names)))

    def test_initialize_rtk_params_present(self):
        m = RDIIModel(n_triangles=2)
        m.initialize()
        for i in (1, 2):
            for prefix in ("R", "T", "K"):
                self.assertIn(f"{prefix}_{i}", m.scalar_parameter_names)

    # --- validate ---

    def test_validate_passes_with_defaults(self):
        m = RDIIModel()
        m.initialize()
        self.assertTrue(m.validate())
        self.assertTrue(m.is_validated())

    def test_validate_fails_R_sum_exceeds_one(self):
        # validate() no longer blocks on R_sum > 1; the constraint is now
        # enforced by the solver via inequality_constraints().
        m = RDIIModel(n_triangles=3)
        m.initialize()
        for i in range(1, 4):
            m._scalar_parameters[f"R_{i}"].value = 0.5
        # validate() should still pass (T_freeze < T_ref, all bounds OK)
        self.assertTrue(m.validate())
        # but inequality_constraints should report a violation
        g = m.inequality_constraints()
        self.assertEqual(len(g), 1)
        self.assertGreater(g[0], 0.0)   # 1.5 - 1.0 = 0.5 > 0 → infeasible

    def test_validate_fails_T_freeze_ge_T_ref(self):
        m = RDIIModel()
        m.initialize()
        m._scalar_parameters["ia_T_freeze"].value = 25.0
        self.assertFalse(m.validate())

    # --- prepare ---

    def test_prepare_infers_dt_hourly(self):
        m = self._prepared()
        self.assertAlmostEqual(m._dt_hours, 1.0)
        self.assertTrue(m.is_prepared())

    def test_prepare_missing_rainfall_column_raises(self):
        m = RDIIModel()
        m.initialize()
        m.validate()
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=5, freq="h"),
        })
        with self.assertRaises(ValueError) as ctx:
            m.prepare(df)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_prepare_missing_datetime_column_raises(self):
        m = RDIIModel()
        m.initialize()
        m.validate()
        df = pd.DataFrame({"rainfall_mm": [1.0, 2.0, 3.0]})
        with self.assertRaises(ValueError) as ctx:
            m.prepare(df)
        self.assertIn("missing required columns", str(ctx.exception))

    def test_prepare_fills_temperature_with_T_ref(self):
        m = RDIIModel(units="metric")
        m.initialize()
        m.validate()
        df = self.sample_df.drop(columns=["temperature_c"])
        m.prepare(df)
        T_ref = m.get_scalar_parameter("ia_T_ref").value
        self.assertTrue((m._prepared_df["temperature_c"] == T_ref).all())

    # --- predict ---

    def test_predict_returns_dataframe(self):
        m = self._prepared()
        result = m.predict()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn("rdii_cfs", result.columns)
        self.assertIn("rdii_mm", result.columns)
        self.assertIn("p_excess_mm", result.columns)
        self.assertIn("datetime", result.columns)

    def test_predict_length_matches_input(self):
        m = self._prepared()
        result = m.predict()
        self.assertEqual(len(result), len(self.sample_df))

    def test_predict_rdii_non_negative(self):
        m = self._prepared()
        result = m.predict()
        self.assertTrue(np.all(result["rdii_mm"].to_numpy() >= -1e-10))
        self.assertTrue(np.all(result["rdii_cfs"].to_numpy() >= -1e-10))

    def test_predict_zero_rain_zero_rdii(self):
        n = 24
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "rainfall_mm": np.zeros(n),
        })
        m = RDIIModel(units="metric")
        m.initialize()
        m.validate()
        m.prepare(df)
        result = m.predict()
        np.testing.assert_allclose(result["rdii_mm"].to_numpy(), 0.0, atol=1e-10)
        np.testing.assert_allclose(result["rdii_cfs"].to_numpy(), 0.0, atol=1e-10)

    def test_predict_state_advances(self):
        m = self._prepared()
        m.predict()
        self.assertTrue(m.is_predicted())

    def test_predict_without_prepare_raises(self):
        m = RDIIModel()
        m.initialize()
        m.validate()
        with self.assertRaises(RuntimeError) as ctx:
            m.predict()
        self.assertIn("prepare", str(ctx.exception))

    def test_predict_respects_parameter_changes(self):
        m = self._prepared()
        result1 = m.predict()
        for i in range(1, 4):
            m._scalar_parameters[f"R_{i}"].value = 0.0
        result2 = m.predict()
        self.assertLess(result2["rdii_cfs"].sum(), result1["rdii_cfs"].sum())

    # --- finalize ---

    def test_finalize_clears_state(self):
        m = self._prepared()
        m.predict()
        m.finalize()
        self.assertTrue(m.is_finalized())
        self.assertIsNone(m._prepared_df)
        self.assertIsNone(m._p_excess)
        self.assertEqual(m._uh_kernels, [])

    # --- full lifecycle ---

    def test_full_lifecycle(self):
        m = RDIIModel(n_triangles=2, units="metric")
        m.initialize()
        self.assertTrue(m.validate())
        m.prepare(self.sample_df)
        result = m.predict()
        m.finalize()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertTrue(m.is_finalized())

    def test_n_triangles_one(self):
        m = self._prepared(n_triangles=1)
        result = m.predict()
        self.assertIn("rdii_cfs", result.columns)
        self.assertIn("rdii_mm", result.columns)

    def test_n_triangles_ten(self):
        m = self._prepared(n_triangles=10)
        result = m.predict()
        self.assertEqual(len(result), len(self.sample_df))

    def test_cfs_volume_conservation_daily(self):
        """CFS totals and depth totals must conserve volume regardless of timestep.

        A single rainfall pulse of known depth should yield the same runoff volume
        whether the model is run at hourly or daily resolution.
        """
        area = 500.0  # acres
        rain_in = 1.0  # inch pulse

        def run(dt_hours):
            dates = pd.date_range("2020-01-01", periods=120, freq=f"{dt_hours}h")
            rain = np.zeros(len(dates))
            rain[5] = rain_in
            df = pd.DataFrame({"datetime": dates, "rainfall_in": rain, "temperature_c": 15.0})
            m = RDIIModel(n_triangles=1, units="imperial")
            m.initialize()
            m.get_scalar_parameter("ia_max").update(value=0.0, lower_bound=0.0)
            m.get_scalar_parameter("ia_k_dep").update(value=0.0, lower_bound=0.0)
            m.get_scalar_parameter("ia_T_freeze").update(value=-5.0)
            m.get_scalar_parameter("R_1").update(value=1.0)
            m.get_scalar_parameter("area_acres").update(value=area)
            m.validate()
            m.prepare(df)
            result = m.predict()
            dt_sec = dt_hours * 3600.0
            cfs_vol = (result["rdii_cfs"] * dt_sec).sum()   # CFS-seconds
            depth_sum = result["rdii_in"].sum()              # inches (depth per step summed)
            return cfs_vol, depth_sum

        vol_hourly, depth_hourly = run(1)
        vol_daily,  depth_daily  = run(24)
        # Both should equal: 1 in * area acres → ft³
        expected_cfs_sec = rain_in / 12.0 * area * 43560.0
        np.testing.assert_allclose(vol_hourly, expected_cfs_sec, rtol=1e-3)
        np.testing.assert_allclose(vol_daily,  expected_cfs_sec, rtol=1e-3)
        # rdii_in is depth per timestep — sums must be equal across resolutions
        np.testing.assert_allclose(depth_hourly, depth_daily, rtol=1e-3)


# ---------------------------------------------------------------------------
# TestRDIICalibration  (replaces old TestRDIIOptimizer; uses generic API)
# ---------------------------------------------------------------------------

@unittest.skipUnless(HAS_PYMOO, "pymoo not installed — skipping calibration tests")
class TestRDIICalibration(unittest.TestCase):

    def setUp(self):
        from sparsehydro.calibration import (
            CalibrationProblem, NSGAIISolver, NashSutcliffe, PeakWeightedMSE,
        )
        self.sample_df = _make_sample_df()
        self.model = _prepared_model()
        obs = np.abs(np.random.default_rng(1).normal(5, 2, len(self.sample_df)))
        self._obs = obs
        self._problem = CalibrationProblem(
            model=self.model,
            objectives=[PeakWeightedMSE(), NashSutcliffe()],
            column_map={
                "observed":  lambda _: obs,
                "predicted": lambda df: df["rdii_cfs"].to_numpy(),
            },
        )
        self._Solver = NSGAIISolver

    def _run(self, pop_size: int = 10, n_gen: int = 3):
        return self._Solver(pop_size=pop_size, n_gen=n_gen, seed=42).solve(
            self._problem
        )

    def test_returns_calibration_result(self):
        from sparsehydro.calibration import CalibrationResult
        self.assertIsInstance(self._run(), CalibrationResult)

    def test_unprepared_model_raises(self):
        from sparsehydro.calibration import CalibrationProblem, PeakWeightedMSE
        m = RDIIModel()
        m.initialize()
        with self.assertRaises(RuntimeError):
            CalibrationProblem(
                model=m,
                objectives=[PeakWeightedMSE()],
                column_map={
                    "observed":  lambda _: np.ones(10),
                    "predicted": lambda df: df["rdii_cfs"].to_numpy(),
                },
            )

    def test_history_length(self):
        result = self._run(n_gen=5)
        self.assertEqual(len(result.history), 5)

    def test_pareto_X_column_count(self):
        result = self._run()
        n_params = len(self.model.scalar_parameter_names)
        self.assertEqual(result.pareto_X.shape[1], n_params)

    def test_objective_names(self):
        result = self._run()
        self.assertIn("peak_weighted_mse", result.objective_names)
        self.assertIn("nash_sutcliffe", result.objective_names)

    def test_best_by_nse_length(self):
        result = self._run()
        best = result.best_by("nash_sutcliffe")
        self.assertEqual(best.shape[0], len(self.model.scalar_parameter_names))

    def test_best_by_pwmse_length(self):
        result = self._run()
        best = result.best_by("peak_weighted_mse")
        self.assertEqual(best.shape[0], len(self.model.scalar_parameter_names))

    def test_to_pareto_dataframe_columns(self):
        result = self._run()
        df = result.to_pareto_dataframe()
        for col in ("generation", "nash_sutcliffe", "peak_weighted_mse", "is_pareto"):
            self.assertIn(col, df.columns)
        for name in result.param_names:
            self.assertIn(name, df.columns)

    def test_to_pareto_dataframe_generation_count(self):
        n_gen = 4
        result = self._run(n_gen=n_gen)
        df = result.to_pareto_dataframe()
        self.assertEqual(df["generation"].nunique(), n_gen)


# ---------------------------------------------------------------------------
# TestVisualization  (skipped when plotly or pymoo is absent)
# ---------------------------------------------------------------------------

@unittest.skipUnless(
    HAS_PLOTLY and HAS_PYMOO,
    "plotly and/or pymoo not installed — skipping visualization tests",
)
class TestVisualization(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration import (
            CalibrationProblem, NSGAIISolver, NashSutcliffe, PeakWeightedMSE,
        )
        cls.sample_df = _make_sample_df()
        cls.model = _prepared_model()
        obs = np.abs(np.random.default_rng(3).normal(3, 1, len(cls.sample_df)))
        problem = CalibrationProblem(
            model=cls.model,
            objectives=[PeakWeightedMSE(), NashSutcliffe()],
            column_map={
                "observed":  lambda _: obs,
                "predicted": lambda df: df["rdii_cfs"].to_numpy(),
            },
        )
        cls.opt_result = NSGAIISolver(pop_size=10, n_gen=4, seed=0).solve(problem)
        cls.predict_result = cls.model.predict()

    def test_plot_timeseries_returns_figure(self):
        import plotly.graph_objects as go
        from sparsehydro.rdii import plot_timeseries
        fig = plot_timeseries(
            datetime=self.sample_df["datetime"],
            rainfall_mm=self.sample_df["rainfall_mm"].to_numpy(),
            observed_flow=self.sample_df["flow_cfs"].to_numpy(),
            predicted_flow=self.predict_result["rdii_mm"].to_numpy(),
        )
        self.assertIsInstance(fig, go.Figure)

    def test_plot_timeseries_no_observed(self):
        import plotly.graph_objects as go
        from sparsehydro.rdii import plot_timeseries
        fig = plot_timeseries(
            datetime=self.sample_df["datetime"],
            rainfall_mm=self.sample_df["rainfall_mm"].to_numpy(),
            observed_flow=None,
            predicted_flow=self.predict_result["rdii_mm"].to_numpy(),
        )
        self.assertIsInstance(fig, go.Figure)

    def test_plot_pareto_evolution_returns_figure(self):
        import plotly.graph_objects as go
        from sparsehydro.rdii import plot_pareto_evolution
        fig = plot_pareto_evolution(self.opt_result)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_pareto_evolution_frame_count(self):
        from sparsehydro.rdii import plot_pareto_evolution
        fig = plot_pareto_evolution(self.opt_result)
        self.assertEqual(len(fig.frames), len(self.opt_result.history))

    def test_plot_parallel_coordinates_returns_figure(self):
        import plotly.graph_objects as go
        from sparsehydro.rdii import plot_parallel_coordinates
        fig = plot_parallel_coordinates(self.opt_result)
        self.assertIsInstance(fig, go.Figure)

    def test_plot_parallel_coordinates_color_by_pwmse(self):
        import plotly.graph_objects as go
        from sparsehydro.rdii import plot_parallel_coordinates
        fig = plot_parallel_coordinates(self.opt_result, color_by="peak_weighted_mse")
        self.assertIsInstance(fig, go.Figure)


if __name__ == "__main__":
    unittest.main()
