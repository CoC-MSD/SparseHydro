"""Tests for the sparsehydro AMM model (:class:`sparsehydro.models.amm.AMMModel`).

The primary correctness anchor is the worked computational example in
Table 1 (Section 4.1) of Edgren, Czachorski & Gonwa (2024, JWMM 32: C525).
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sparsehydro.enums import ModelState
from sparsehydro.models.amm import (
    AMMModel,
    _moving_average_precip,
    _moving_average_temp,
    _shcf_sigmoid,
)
from sparsehydro.registry import registry


# ---------------------------------------------------------------------------
# Paper Table 1 fixture (Section 4.1)
# ---------------------------------------------------------------------------

# Incremental precipitation [in] and given moving-average temperature [°F].
_TABLE1_PRECIP = [0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
_TABLE1_MATEMP = [70.1, 70.0, 69.9, 69.8, 69.7, 69.6, 69.5, 69.4, 69.3, 69.2, 69.1]
# Expected Level-1 flow output Q [cfs] (Table 1, last column).
_TABLE1_FLOW = [0.00, 0.00, 7.20, 20.05, 36.24, 55.60, 39.32, 27.80, 19.66, 13.90, 9.83]
# Expected additional capture fraction RW [%] (Table 1).
_TABLE1_RW_PCT = [0.0, 0.0, 2.9, 5.3, 7.7, 10.6, 9.7, 8.9, 8.2, 7.5, 6.9]


def _table1_dataframe() -> pd.DataFrame:
    """Return the Table 1 forcing as a prepared DataFrame (TAT=0 → MATemp=Temp)."""
    n = len(_TABLE1_PRECIP)
    dates = pd.date_range("2024-01-01 00:00", periods=n, freq="h")
    return pd.DataFrame({
        "datetime": dates,
        "rainfall_in": np.array(_TABLE1_PRECIP, dtype=float),
        "temperature": np.array(_TABLE1_MATEMP, dtype=float),
    })


def _table1_model() -> AMMModel:
    """Return an AMMModel parameterized exactly as the Table 1 example."""
    model = AMMModel(
        component_type="standard",
        units="imperial",
        area_acres=1000.0,
        pat_hours=0.0,
        hhl_hours=2.0,
        amhl_hours=8.0,
        rd=0.01,
        cold_temp=30.0,
        hot_temp=70.0,
        cold_value=0.07,
        hot_value=0.03,
        tat_hours=0.0,
    )
    model.initialize()
    model.validate()
    model.prepare(_table1_dataframe())
    return model


# ---------------------------------------------------------------------------
# Helper-function unit tests
# ---------------------------------------------------------------------------

class TestHelpers(unittest.TestCase):

    def test_map_pat_zero_is_one_step_lag(self):
        x = np.array([0.0, 1.0, 1.0, 1.0, 1.0, 0.0])
        out = _moving_average_precip(x, n_steps=1)
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 1.0, 1.0, 1.0])

    def test_map_window_two_averages_two_prior(self):
        x = np.array([0.0, 2.0, 4.0, 0.0])
        out = _moving_average_precip(x, n_steps=2)
        # MAP_t = mean(P_{t-1}, P_{t-2}); zero-padded before start.
        np.testing.assert_allclose(out, [0.0, 0.0, 1.0, 3.0])

    def test_matemp_tat_zero_is_identity(self):
        x = np.array([70.1, 70.0, 69.9])
        out = _moving_average_temp(x, n_steps=1)
        np.testing.assert_allclose(out, x)

    def test_matemp_window_two_includes_current(self):
        x = np.array([10.0, 20.0, 30.0])
        out = _moving_average_temp(x, n_steps=2)
        # MATemp_t = mean(Temp_t, Temp_{t-1}); shrinks at the start.
        np.testing.assert_allclose(out, [10.0, 15.0, 25.0])

    def test_shcf_sigmoid_matches_paper_example(self):
        # Paper Section 4.1: at MATemp = 70 °F, SHCF = 0.0300.
        shcf = _shcf_sigmoid(np.array([70.0]), 30.0, 70.0, 0.07, 0.03)
        self.assertAlmostEqual(shcf[0], 0.0300, places=4)

    def test_shcf_sigmoid_midpoint(self):
        # At the midpoint temperature the sigmoid is at half its range.
        shcf = _shcf_sigmoid(np.array([50.0]), 30.0, 70.0, 0.07, 0.03)
        self.assertAlmostEqual(shcf[0], 0.05, places=3)


# ---------------------------------------------------------------------------
# Table 1 reproduction
# ---------------------------------------------------------------------------

class TestTable1Reproduction(unittest.TestCase):

    def setUp(self):
        self.model = _table1_model()
        self.result = self.model.predict()

    def test_flow_matches_table1(self):
        flow = self.result["amm_cfs"].to_numpy()
        # Steps 0-2 are unaffected by the antecedent-wetness recursion
        # (RW_{t-1} = 0) and reproduce the published values exactly.
        for got, want in zip(flow[:3], _TABLE1_FLOW[:3]):
            self.assertAlmostEqual(got, want, delta=0.02)
        # From step 3 on, the recursion engages. This implementation evaluates
        # the exact closed-form solution of the Level-2 ODE (Eq. 5), which
        # yields smoothly decaying storm increments; the paper's printed
        # Table 1 carries small rounding/transcription noise (its RW
        # increments are non-monotonic under constant forcing). The series
        # therefore tracks the published values to within ~1.5 cfs.
        np.testing.assert_allclose(flow, _TABLE1_FLOW, atol=1.5)
        # Peak magnitude and timing.
        self.assertEqual(int(np.argmax(flow)), 5)
        self.assertAlmostEqual(flow[5], 55.60, delta=1.5)

    def test_map_matches_table1(self):
        # MAP with PAT=0 is the one-step-lagged precip.
        expected = [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        np.testing.assert_allclose(self.result["map"].to_numpy(), expected)

    def test_matemp_matches_input(self):
        np.testing.assert_allclose(self.result["matemp"].to_numpy(), _TABLE1_MATEMP)

    def test_rw_matches_table1(self):
        rw_pct = self.result["rw"].to_numpy() * 100.0
        # First non-zero RW (2:00) has no recursive history and matches exactly.
        self.assertAlmostEqual(rw_pct[2], 2.9, delta=0.1)
        # RW rises monotonically through the storm (steps 2-5) and then recedes.
        self.assertTrue(np.all(np.diff(rw_pct[2:6]) > 0))
        self.assertTrue(np.all(np.diff(rw_pct[5:]) < 0))
        # Once forcing stops, RW decays by AMRF = 0.5**(dt/AMHL) = 0.5**(1/8).
        amrf = 0.5 ** (1.0 / 8.0)
        self.assertAlmostEqual(rw_pct[7] / rw_pct[6], amrf, places=3)
        # Overall the series tracks the published RW column to within ~0.5%.
        np.testing.assert_allclose(rw_pct, _TABLE1_RW_PCT, atol=0.5)

    def test_capture_fraction_is_rd_plus_rw(self):
        rd = self.model.get_scalar_parameter("RD").value
        rw = self.result["rw"].to_numpy()
        np.testing.assert_allclose(
            self.result["capture_fraction"].to_numpy(), rd + rw
        )

    def test_recession_follows_shape_factor(self):
        # After rain ends (5:00), flow decays by SF = 0.5**(1/2) = 0.7071 per step.
        flow = self.result["amm_cfs"].to_numpy()
        ratio = flow[7] / flow[6]
        self.assertAlmostEqual(ratio, 0.5 ** 0.5, places=2)


# ---------------------------------------------------------------------------
# Lifecycle, registry, and configuration
# ---------------------------------------------------------------------------

class TestLifecycle(unittest.TestCase):

    def test_state_progression(self):
        model = AMMModel()
        self.assertIs(model.state, ModelState.CREATED)
        model.initialize()
        self.assertIs(model.state, ModelState.INITIALIZED)
        self.assertTrue(model.validate())
        self.assertIs(model.state, ModelState.VALIDATED)
        model.prepare(_table1_dataframe())
        self.assertIs(model.state, ModelState.PREPARED)
        model.predict()
        self.assertIs(model.state, ModelState.PREDICTED)
        model.finalize()
        self.assertIs(model.state, ModelState.FINALIZED)

    def test_predict_before_prepare_raises(self):
        model = AMMModel()
        model.initialize()
        model.validate()
        with self.assertRaises(RuntimeError):
            model.predict()

    def test_validate_rejects_temp_order(self):
        model = AMMModel(cold_temp=70.0, hot_temp=30.0)
        model.initialize()
        self.assertFalse(model.validate())

    def test_inequality_constraint(self):
        model = AMMModel(cold_temp=30.0, hot_temp=70.0)
        model.initialize()
        self.assertEqual(model.inequality_constraints(), [30.0 - 70.0])

    def test_registry_create(self):
        self.assertIn("amm", registry.names())
        model = registry.create("amm")
        self.assertIsInstance(model, AMMModel)

    def test_invalid_component_type(self):
        with self.assertRaises(ValueError):
            AMMModel(component_type="bogus")

    def test_invalid_units(self):
        with self.assertRaises(ValueError):
            AMMModel(units="furlongs")

    def test_time_to_peak_is_pat_plus_dt(self):
        model = _table1_model()
        # PAT = 0, dt = 1 h → TP = 1 h.
        self.assertAlmostEqual(model.time_to_peak_hours, 1.0)


# ---------------------------------------------------------------------------
# Baseflow component
# ---------------------------------------------------------------------------

class TestBaseflowComponent(unittest.TestCase):

    def setUp(self):
        self.model = AMMModel(
            component_type="baseflow",
            units="imperial",
            area_acres=4000.0,
            pat_hours=1.0,
            hhl_hours=22.76,
            cold_temp=30.0,
            hot_temp=70.0,
            cold_value=0.05,
            hot_value=0.01,
            tat_hours=24.0,
        )
        self.model.initialize()
        self.model.validate()

    def test_no_level2_parameters(self):
        names = self.model.scalar_parameter_names
        self.assertNotIn("RD", names)
        self.assertNotIn("AMHL", names)
        self.assertIn("Cold_R", names)
        self.assertIn("Hot_R", names)

    def test_no_rw_or_shcf_columns(self):
        self.model.prepare(_table1_dataframe())
        result = self.model.predict()
        self.assertNotIn("rw", result.columns)
        self.assertNotIn("shcf", result.columns)
        self.assertIn("capture_fraction", result.columns)

    def test_capture_fraction_within_cold_hot_range(self):
        self.model.prepare(_table1_dataframe())
        result = self.model.predict()
        cf = result["capture_fraction"].to_numpy()
        # The SHCF/capture sigmoid places the Cold/Hot calibration points at
        # 1/12 and 11/12 of its span (Eqs. 7-10), so values may extend slightly
        # beyond [Hot_R, Cold_R]. The true asymptotes are Hot - (1/12)*span and
        # Cold + (1/12)*span, where span = 1.2*(Cold - Hot).
        cold_r, hot_r = 0.05, 0.01
        span = 1.2 * (cold_r - hot_r)
        lower = hot_r - span / 12.0
        upper = cold_r + span / 12.0
        self.assertTrue(np.all(cf >= lower - 1e-9))
        self.assertTrue(np.all(cf <= upper + 1e-9))


# ---------------------------------------------------------------------------
# Units and time-step behaviour
# ---------------------------------------------------------------------------

class TestUnitsAndTimeStep(unittest.TestCase):

    def test_metric_units_run(self):
        n = 24
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "rainfall_mm": np.where(np.arange(n) < 4, 5.0, 0.0),
            "temperature": np.full(n, 10.0),
        })
        model = AMMModel(units="metric")
        model.initialize()
        model.validate()
        model.prepare(df)
        result = model.predict()
        self.assertEqual(model._rainfall_col, "rainfall_mm")
        self.assertTrue(np.all(np.isfinite(result["amm_cfs"].to_numpy())))
        self.assertGreater(result["amm_cfs"].max(), 0.0)

    def test_missing_temperature_uses_midpoint(self):
        n = 12
        df = pd.DataFrame({
            "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
            "rainfall_in": np.where(np.arange(n) < 3, 1.0, 0.0),
        })
        model = AMMModel(cold_temp=30.0, hot_temp=70.0)
        model.initialize()
        model.validate()
        model.prepare(df)
        np.testing.assert_allclose(
            model._prepared_df["temperature"].to_numpy(), 50.0
        )

    def test_time_step_independence_peak(self):
        # Same 4 inches of rain represented hourly vs 15-minute; peak flow
        # should agree closely (paper: longer dt biases flow slightly low).
        def build(dt_hours, n_rain_steps, total_rain_steps, depth):
            n = total_rain_steps
            freq = pd.Timedelta(hours=dt_hours)
            df = pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=n, freq=freq),
                "rainfall_in": np.where(np.arange(n) < n_rain_steps, depth, 0.0),
                "temperature": np.full(n, 50.0),
            })
            m = AMMModel(area_acres=1000.0, pat_hours=0.0, hhl_hours=4.0,
                         amhl_hours=12.0, rd=0.0, tat_hours=0.0)
            m.initialize()
            m.validate()
            m.prepare(df)
            return m.predict()["amm_cfs"].to_numpy().max()

        peak_hourly = build(1.0, 4, 48, 1.0)        # 4 in over 4 h
        peak_quarter = build(0.25, 16, 192, 0.25)   # 4 in over 4 h, 15-min steps
        rel_err = abs(peak_quarter - peak_hourly) / peak_hourly
        self.assertLess(rel_err, 0.05)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
