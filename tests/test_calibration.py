"""Tests for sparsehydro.calibration — unittest.TestCase.

Covers:
  - IObjective subclasses (MSE, RMSE, MAE, PeakWeightedMSE, NashSutcliffe, KGE)
  - CalibrationProblem (construction, evaluate, make_copy)
  - CalibrationResult (best_by, objective_display_values, to_pareto_dataframe)
  - _identify_pareto helper
  - NSGAIISolver (conditional on pymoo)
  - ScipySolver (conditional on scipy)
"""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from sparsehydro.calibration import (
    IObjective,
    KGE,
    MAE,
    MSE,
    RMSE,
    CalibrationProblem,
    CalibrationResult,
    GenerationRecord,
    NashSutcliffe,
    PeakWeightedMSE,
)
from sparsehydro.calibration.result import _identify_pareto
from sparsehydro.enums import ModelState
from sparsehydro.interfaces import IModel
from sparsehydro.parameters import ScalarParameter

# ---------------------------------------------------------------------------
# Optional-dependency flags
# ---------------------------------------------------------------------------

try:
    import pymoo  # noqa: F401
    HAS_PYMOO = True
except ImportError:
    HAS_PYMOO = False

try:
    import scipy  # noqa: F401
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Minimal test model (no RDII dependency)
# ---------------------------------------------------------------------------

class _LinearModel(IModel):
    """y = slope * x + intercept  — for testing CalibrationProblem."""

    model_name = "test_linear_calib"

    def initialize(self) -> None:
        self.register_scalar_parameter(
            ScalarParameter("slope", value=1.0, lower_bound=0.0, upper_bound=10.0)
        )
        self.register_scalar_parameter(
            ScalarParameter("intercept", value=0.0, lower_bound=-5.0, upper_bound=5.0)
        )
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        if self.parameters_valid():
            self._state = ModelState.VALIDATED
            return True
        return False

    def prepare(self, x: np.ndarray, **kwargs) -> None:
        self._x = np.asarray(x, dtype=float)
        self._state = ModelState.PREPARED

    def predict(self, **kwargs) -> pd.DataFrame:
        slope = self.get_scalar_parameter("slope").value
        intercept = self.get_scalar_parameter("intercept").value
        y = slope * self._x + intercept
        self._state = ModelState.PREDICTED
        return pd.DataFrame({"y": y})

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


def _make_linear_model_and_problem():
    """Return a CalibrationProblem around a prepared _LinearModel."""
    x = np.linspace(0, 10, 20)
    true_slope, true_intercept = 2.5, 1.0
    observed = true_slope * x + true_intercept + np.random.default_rng(0).normal(0, 0.1, len(x))

    model = _LinearModel()
    model.initialize()
    model.validate()
    model.prepare(x)

    problem = CalibrationProblem(
        model=model,
        observed=observed,
        objectives=[MSE(), NashSutcliffe()],
        result_extractor=lambda df: df["y"].to_numpy(),
    )
    return problem, observed, true_slope, true_intercept


# ===========================================================================
# Test IObjective subclasses
# ===========================================================================

class TestObjectives(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(1)
        self.obs = rng.uniform(1, 10, 50)
        self.pred = rng.uniform(1, 10, 50)

    # --- shape mismatch ---

    def _assert_shape_mismatch(self, obj: IObjective):
        with self.assertRaises(Exception):
            obj.evaluate(np.array([1.0, 2.0]), np.array([1.0]))

    def test_mse_shape_mismatch(self):
        self._assert_shape_mismatch(MSE())

    def test_rmse_shape_mismatch(self):
        self._assert_shape_mismatch(RMSE())

    def test_mae_shape_mismatch(self):
        self._assert_shape_mismatch(MAE())

    # --- perfect predictor ---

    def test_mse_perfect(self):
        self.assertAlmostEqual(MSE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_rmse_perfect(self):
        self.assertAlmostEqual(RMSE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_mae_perfect(self):
        self.assertAlmostEqual(MAE().evaluate(self.obs, self.obs), 0.0, places=12)

    def test_nse_perfect(self):
        self.assertAlmostEqual(NashSutcliffe().evaluate(self.obs, self.obs), 1.0, places=12)

    def test_kge_perfect(self):
        self.assertAlmostEqual(KGE().evaluate(self.obs, self.obs), 1.0, places=10)

    # --- minimize flags ---

    def test_mse_minimize(self):
        self.assertTrue(MSE.minimize)

    def test_rmse_minimize(self):
        self.assertTrue(RMSE.minimize)

    def test_mae_minimize(self):
        self.assertTrue(MAE.minimize)

    def test_peak_weighted_mse_minimize(self):
        self.assertTrue(PeakWeightedMSE.minimize)

    def test_nse_maximize(self):
        self.assertFalse(NashSutcliffe.minimize)

    def test_kge_maximize(self):
        self.assertFalse(KGE.minimize)

    # --- MSE / RMSE relationship ---

    def test_rmse_is_sqrt_mse(self):
        mse_val = MSE().evaluate(self.obs, self.pred)
        rmse_val = RMSE().evaluate(self.obs, self.pred)
        self.assertAlmostEqual(rmse_val, mse_val ** 0.5, places=10)

    # --- NSE constant observed raises ---

    def test_nse_constant_obs_raises(self):
        obs = np.ones(10)
        pred = np.ones(10) * 1.5
        with self.assertRaises(ValueError):
            NashSutcliffe().evaluate(obs, pred)

    # --- KGE zero std / zero mean raises ---

    def test_kge_zero_std_raises(self):
        obs = np.ones(10)
        with self.assertRaises(ValueError):
            KGE().evaluate(obs, obs)

    # --- names ---

    def test_objective_names(self):
        self.assertEqual(MSE.name, "mse")
        self.assertEqual(RMSE.name, "rmse")
        self.assertEqual(MAE.name, "mae")
        self.assertEqual(NashSutcliffe.name, "nash_sutcliffe")
        self.assertEqual(KGE.name, "kge")
        self.assertEqual(PeakWeightedMSE.name, "peak_weighted_mse")

    # --- PeakWeightedMSE correct weighting ---

    def test_peak_weighted_mse_perfect(self):
        obs = np.array([1.0, 2.0, 5.0, 1.0])
        self.assertAlmostEqual(PeakWeightedMSE().evaluate(obs, obs), 0.0, places=12)

    def test_peak_weighted_mse_positive(self):
        obs = np.array([1.0, 3.0, 5.0, 2.0])
        pred = np.array([1.0, 2.0, 4.0, 2.5])
        self.assertGreater(PeakWeightedMSE().evaluate(obs, pred), 0.0)


# ===========================================================================
# Test _identify_pareto
# ===========================================================================

class TestIdentifyPareto(unittest.TestCase):

    def test_single_solution_is_pareto(self):
        F = np.array([[1.0, 2.0]])
        mask = _identify_pareto(F)
        self.assertEqual(mask.tolist(), [True])

    def test_all_pareto_when_tradeoff(self):
        F = np.array([[0.0, 1.0], [1.0, 0.0]])
        mask = _identify_pareto(F)
        self.assertTrue(all(mask))

    def test_dominated_solution_excluded(self):
        F = np.array([
            [1.0, 1.0],
            [2.0, 2.0],  # dominated by first
            [0.5, 1.5],
        ])
        mask = _identify_pareto(F)
        self.assertFalse(mask[1])
        self.assertTrue(mask[0])
        self.assertTrue(mask[2])

    def test_all_identical_all_pareto(self):
        F = np.array([[1.0, 1.0], [1.0, 1.0]])
        mask = _identify_pareto(F)
        self.assertTrue(all(mask))


# ===========================================================================
# Test CalibrationResult
# ===========================================================================

class TestCalibrationResult(unittest.TestCase):

    def _make_result(self) -> CalibrationResult:
        """Two-objective result: obj0 minimised, obj1 maximised (stored negated)."""
        pareto_X = np.array([[1.0, 2.0], [3.0, 4.0]])
        pareto_F = np.array([[0.5, -0.8], [0.3, -0.9]])  # obj1 negated (maximised)
        rec = GenerationRecord(
            generation=1,
            X=pareto_X.copy(),
            F=pareto_F.copy(),
            n_pareto=2,
        )
        return CalibrationResult(
            history=[rec],
            pareto_X=pareto_X,
            pareto_F=pareto_F,
            param_names=["p0", "p1"],
            objective_names=["mse", "nse"],
            minimize_flags=[True, False],
        )

    def test_objective_display_values_unnegates_maximised(self):
        result = self._make_result()
        display = result.objective_display_values()
        # obj0 (minimise): no change → 0.5, 0.3
        np.testing.assert_allclose(display[:, 0], [0.5, 0.3])
        # obj1 (maximise): negated back → 0.8, 0.9
        np.testing.assert_allclose(display[:, 1], [0.8, 0.9])

    def test_best_by_minimised_objective(self):
        result = self._make_result()
        best = result.best_by("mse")
        # pareto_F col0: [0.5, 0.3] → argmin = 1 → pareto_X[1] = [3, 4]
        np.testing.assert_array_equal(best, [3.0, 4.0])

    def test_best_by_maximised_objective(self):
        result = self._make_result()
        best = result.best_by("nse")
        # pareto_F col1: [-0.8, -0.9] → argmin = 1 → pareto_X[1] = [3, 4]
        np.testing.assert_array_equal(best, [3.0, 4.0])

    def test_best_by_invalid_name_raises(self):
        result = self._make_result()
        with self.assertRaises(ValueError):
            result.best_by("nonexistent")

    def test_to_pareto_dataframe_columns(self):
        result = self._make_result()
        df = result.to_pareto_dataframe()
        expected = {"generation", "p0", "p1", "mse", "nse", "is_pareto"}
        self.assertTrue(expected.issubset(set(df.columns)))

    def test_to_pareto_dataframe_display_form(self):
        result = self._make_result()
        df = result.to_pareto_dataframe()
        pareto_rows = df[df["is_pareto"]]
        # NSE column should be un-negated: 0.8 and 0.9
        self.assertTrue((pareto_rows["nse"] > 0).all())

    def test_to_pareto_dataframe_is_pareto_only_in_final_gen(self):
        pareto_X = np.array([[1.0, 2.0]])
        pareto_F = np.array([[0.5, -0.8]])
        recs = [
            GenerationRecord(1, pareto_X.copy(), pareto_F.copy(), 1),
            GenerationRecord(2, pareto_X.copy(), pareto_F.copy(), 1),
        ]
        result = CalibrationResult(
            history=recs,
            pareto_X=pareto_X,
            pareto_F=pareto_F,
            param_names=["p0", "p1"],
            objective_names=["mse"],
            minimize_flags=[True],
        )
        df = result.to_pareto_dataframe()
        gen1_pareto = df[df["generation"] == 1]["is_pareto"].tolist()
        gen2_pareto = df[df["generation"] == 2]["is_pareto"].tolist()
        self.assertFalse(any(gen1_pareto))
        self.assertTrue(any(gen2_pareto))


# ===========================================================================
# Test CalibrationProblem
# ===========================================================================

class TestCalibrationProblem(unittest.TestCase):

    def setUp(self):
        self.problem, self.observed, self.slope, self.intercept = \
            _make_linear_model_and_problem()

    def test_n_params(self):
        self.assertEqual(self.problem.n_params, 2)

    def test_n_objectives(self):
        self.assertEqual(self.problem.n_objectives, 2)

    def test_param_names(self):
        self.assertEqual(self.problem.param_names, ["slope", "intercept"])

    def test_objective_names(self):
        self.assertEqual(self.problem.objective_names, ["mse", "nash_sutcliffe"])

    def test_minimize_flags(self):
        self.assertEqual(self.problem.minimize_flags, [True, False])

    def test_bounds_shape(self):
        xl, xu = self.problem.bounds
        self.assertEqual(xl.shape, (2,))
        self.assertEqual(xu.shape, (2,))

    def test_evaluate_returns_correct_length(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        self.assertEqual(F.shape, (2,))

    def test_evaluate_mse_near_zero_for_true_params(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        self.assertLess(F[0], 0.1)  # MSE ~ 0 for near-true params

    def test_evaluate_nse_negated(self):
        x = np.array([self.slope, self.intercept])
        F = self.problem.evaluate(x)
        # F[1] should be negative (NSE≈1 negated → ≈-1)
        self.assertLess(F[1], 0.0)

    def test_evaluate_updates_parameters(self):
        x = np.array([5.0, 2.5])
        self.problem.evaluate(x)
        self.assertAlmostEqual(
            self.problem._model.get_scalar_parameter("slope").value, 5.0
        )
        self.assertAlmostEqual(
            self.problem._model.get_scalar_parameter("intercept").value, 2.5
        )

    def test_make_copy_is_independent(self):
        copy = self.problem.make_copy()
        copy._model.get_scalar_parameter("slope").value = 99.0
        # Original unchanged
        self.assertNotAlmostEqual(
            self.problem._model.get_scalar_parameter("slope").value, 99.0
        )

    def test_make_copy_is_prepared(self):
        copy = self.problem.make_copy()
        self.assertTrue(copy._model.is_prepared())

    def test_model_not_prepared_raises(self):
        m = _LinearModel()
        m.initialize()
        m.validate()
        # not prepared
        with self.assertRaises(RuntimeError):
            CalibrationProblem(
                model=m,
                observed=np.ones(5),
                objectives=[MSE()],
                result_extractor=lambda df: df["y"].to_numpy(),
            )

    def test_empty_objectives_raises(self):
        x = np.linspace(0, 5, 10)
        m = _LinearModel()
        m.initialize()
        m.validate()
        m.prepare(x)
        with self.assertRaises(ValueError):
            CalibrationProblem(
                model=m,
                observed=np.ones(10),
                objectives=[],
                result_extractor=lambda df: df["y"].to_numpy(),
            )


# ===========================================================================
# Test NSGAIISolver (conditional)
# ===========================================================================

@unittest.skipUnless(HAS_PYMOO, "pymoo not installed")
class TestNSGAIISolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.nsga2 import NSGAIISolver
        cls.NSGAIISolver = NSGAIISolver
        cls.problem, cls.observed, cls.slope, cls.intercept = \
            _make_linear_model_and_problem()

    def test_solve_returns_calibration_result(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_result_has_history(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(len(result.history), 5)

    def test_result_pareto_X_shape(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.pareto_X.shape[1], 2)

    def test_result_objective_names_preserved(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])

    def test_result_minimize_flags_preserved(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.minimize_flags, [True, False])

    def test_best_by_mse_valid_shape(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        best = result.best_by("mse")
        self.assertEqual(best.shape, (2,))

    def test_convergence_direction(self):
        solver = self.NSGAIISolver(pop_size=30, n_gen=20, seed=42)
        result = solver.solve(self.problem)
        early_mse = result.history[0].F[:, 0].min()
        late_mse = result.history[-1].F[:, 0].min()
        self.assertLessEqual(late_mse, early_mse + 1e-6)

    def test_to_pareto_dataframe_has_all_objectives(self):
        solver = self.NSGAIISolver(pop_size=20, n_gen=5, seed=0)
        result = solver.solve(self.problem)
        df = result.to_pareto_dataframe()
        self.assertIn("mse", df.columns)
        self.assertIn("nash_sutcliffe", df.columns)


# ===========================================================================
# Test ScipySolver (conditional)
# ===========================================================================

@unittest.skipUnless(HAS_SCIPY, "scipy not installed")
class TestScipySolver(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        from sparsehydro.calibration.solvers.scipy_solver import ScipySolver
        cls.ScipySolver = ScipySolver
        cls.problem, cls.observed, cls.slope, cls.intercept = \
            _make_linear_model_and_problem()

    def test_solve_returns_calibration_result(self):
        solver = self.ScipySolver(method="differential_evolution",
                                  objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)

    def test_result_single_pareto_solution(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.pareto_X.shape[0], 1)

    def test_result_objective_names_preserved(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.objective_names, ["mse", "nash_sutcliffe"])

    def test_result_minimize_flags_preserved(self):
        solver = self.ScipySolver(objective_index=0, maxiter=50, seed=0)
        result = solver.solve(self.problem)
        self.assertEqual(result.minimize_flags, [True, False])

    def test_mse_converges_toward_zero(self):
        solver = self.ScipySolver(objective_index=0, maxiter=200, seed=42)
        result = solver.solve(self.problem)
        best = result.best_by("mse")
        # Evaluate MSE at best params
        F = self.problem.make_copy().evaluate(best)
        self.assertLess(F[0], 1.0)

    def test_scipy_minimize_method(self):
        solver = self.ScipySolver(method="L-BFGS-B", objective_index=0, maxiter=100)
        result = solver.solve(self.problem)
        self.assertIsInstance(result, CalibrationResult)
        self.assertEqual(result.pareto_X.shape[0], 1)


if __name__ == "__main__":
    unittest.main()
