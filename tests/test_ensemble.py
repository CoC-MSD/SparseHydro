"""Tests for EnsembleModel — generic multi-model compositor."""

import unittest
from typing import Any

import numpy as np
import pandas as pd

from sparsehydro.models import EnsembleModel
from sparsehydro.enums import ModelState
from sparsehydro.interfaces import IModel
from sparsehydro.parameters import ConstraintRecord, ScalarParameter


# ---------------------------------------------------------------------------
# Minimal toy model for testing
# ---------------------------------------------------------------------------

class ScaleModel(IModel):
    """A trivial model: output = scale * sum(input_col)."""

    model_name = "scale_model"

    def __init__(self, input_col: str = "x", output_col: str = "y") -> None:
        super().__init__()
        self._input_col = input_col
        self._output_col = output_col
        self._data: pd.DataFrame | None = None

    def initialize(self) -> None:
        self.register_scalar_parameter(ScalarParameter(
            "scale", value=1.0, lower_bound=0.0, upper_bound=10.0, units="-",
            description="Scaling factor",
        ))
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        if not self.parameters_valid():
            return False
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, data: Any, **kwargs: Any) -> None:
        self._data = data
        self._state = ModelState.PREPARED

    def predict(self) -> pd.DataFrame:
        scale = self.get_scalar_parameter("scale").value
        x = self._data[self._input_col].to_numpy(dtype=float)
        self._state = ModelState.PREDICTED
        return pd.DataFrame({
            "datetime": self._data["datetime"].to_numpy(),
            self._output_col: x * scale,
        })

    def finalize(self) -> None:
        self._data = None
        self._state = ModelState.FINALIZED


class ConstrainedModel(ScaleModel):
    """ScaleModel that also has an inequality constraint: scale <= 5."""

    model_name = "constrained_model"

    def initialize(self) -> None:
        super().initialize()
        self.register_inequality_constraint(ConstraintRecord(
            name="scale_leq_5",
            description="scale parameter ≤ 5.0",
        ))

    def inequality_constraints(self) -> list[float]:
        return [self.get_scalar_parameter("scale").value - 5.0]


def _make_data(n: int = 10) -> pd.DataFrame:
    t = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({"datetime": t, "x": np.arange(1, n + 1, dtype=float)})


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEnsembleModelConstruction(unittest.TestCase):

    def test_invalid_mode_raises(self):
        m = ScaleModel()
        with self.assertRaises(ValueError):
            EnsembleModel([(m, lambda df: df["y"].to_numpy())], mode="invalid")

    def test_empty_components_raises(self):
        with self.assertRaises(ValueError):
            EnsembleModel([])

    def test_alias_length_mismatch_raises(self):
        m = ScaleModel()
        with self.assertRaises(ValueError):
            EnsembleModel(
                [(m, lambda df: df["y"].to_numpy())],
                aliases=["a", "b"],  # 2 aliases for 1 component
            )

    def test_default_aliases(self):
        m1, m2 = ScaleModel(), ScaleModel()
        e = EnsembleModel([
            (m1, lambda df: df["y"].to_numpy()),
            (m2, lambda df: df["y"].to_numpy()),
        ])
        self.assertEqual(e.aliases, ["model_1", "model_2"])

    def test_custom_aliases(self):
        m1, m2 = ScaleModel(), ScaleModel()
        e = EnsembleModel(
            [(m1, lambda df: df["y"].to_numpy()), (m2, lambda df: df["y"].to_numpy())],
            aliases=["alpha", "beta"],
        )
        self.assertEqual(e.aliases, ["alpha", "beta"])

    def test_properties(self):
        m = ScaleModel()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], mode="product",
                          output_name="combined")
        self.assertEqual(e.mode, "product")
        self.assertEqual(e.output_name, "combined")
        self.assertEqual(e.n_components, 1)


class TestEnsembleModelInitialize(unittest.TestCase):

    def setUp(self):
        self.m1 = ScaleModel(input_col="x", output_col="y")
        self.m2 = ScaleModel(input_col="x", output_col="y")
        self.e = EnsembleModel(
            [(self.m1, lambda df: df["y"].to_numpy()),
             (self.m2, lambda df: df["y"].to_numpy())],
            aliases=["a", "b"],
        )
        self.e.initialize()

    def test_children_auto_initialized(self):
        self.assertTrue(self.m1.is_validated())
        self.assertTrue(self.m2.is_validated())

    def test_weight_parameters_registered(self):
        names = self.e.scalar_parameter_names
        self.assertIn("w_1", names)
        self.assertIn("w_2", names)

    def test_child_params_prefixed(self):
        names = self.e.scalar_parameter_names
        self.assertIn("a_scale", names)
        self.assertIn("b_scale", names)

    def test_total_parameter_count(self):
        # w_1, w_2, a_scale, b_scale = 4
        self.assertEqual(len(self.e.scalar_parameter_names), 4)

    def test_weight_defaults_sum_mode(self):
        # Default weight = 1/N = 0.5 for 2 components
        self.assertAlmostEqual(self.e.get_scalar_parameter("w_1").value, 0.5)
        self.assertAlmostEqual(self.e.get_scalar_parameter("w_2").value, 0.5)

    def test_sum_w_constraint_registered(self):
        self.assertIn("sum_w_leq_1", self.e.inequality_constraint_names)

    def test_no_weight_constraint_product_mode(self):
        m = ScaleModel()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], mode="product")
        e.initialize()
        self.assertNotIn("sum_w_leq_1", e.inequality_constraint_names)

    def test_state_initialized(self):
        self.assertTrue(self.e.is_initialized())


class TestEnsembleModelConstraintSurfacing(unittest.TestCase):

    def test_child_constraints_surfaced_with_prefix(self):
        cm1 = ConstrainedModel(input_col="x", output_col="y")
        cm2 = ConstrainedModel(input_col="x", output_col="y")
        e = EnsembleModel(
            [(cm1, lambda df: df["y"].to_numpy()),
             (cm2, lambda df: df["y"].to_numpy())],
            aliases=["p", "q"],
        )
        e.initialize()
        names = e.inequality_constraint_names
        self.assertIn("p_scale_leq_5", names)
        self.assertIn("q_scale_leq_5", names)
        self.assertIn("sum_w_leq_1", names)

    def test_inequality_constraints_residuals(self):
        cm = ConstrainedModel(input_col="x", output_col="y")
        e = EnsembleModel([(cm, lambda df: df["y"].to_numpy())], aliases=["m"])
        e.initialize()
        e.validate()
        data = _make_data()
        e.prepare(data)
        g = e.inequality_constraints()
        # child: scale=1.0 → residual = 1.0 - 5.0 = -4.0 (feasible)
        self.assertAlmostEqual(g[0], 1.0 - 5.0)
        # ensemble: w_1 = 1.0 → residual = 1.0 - 1.0 = 0.0 (exactly on boundary)
        self.assertAlmostEqual(g[1], 0.0)

    def test_no_child_constraints_unconstrained_model(self):
        m = ScaleModel()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], aliases=["m"])
        e.initialize()
        # Only sum_w_leq_1
        self.assertEqual(e.inequality_constraint_names, ["sum_w_leq_1"])


class TestEnsembleSumMode(unittest.TestCase):

    def setUp(self):
        self.m1 = ScaleModel(input_col="x", output_col="y")
        self.m2 = ScaleModel(input_col="x", output_col="y")
        self.e = EnsembleModel(
            [(self.m1, lambda df: df["y"].to_numpy()),
             (self.m2, lambda df: df["y"].to_numpy())],
            aliases=["a", "b"],
            output_name="combined",
        )
        self.e.initialize()
        self.e.validate()
        self.data = _make_data()
        self.e.prepare(self.data)

    def test_predict_returns_dataframe(self):
        out = self.e.predict()
        self.assertIsInstance(out, pd.DataFrame)

    def test_predict_columns(self):
        out = self.e.predict()
        self.assertIn("datetime", out.columns)
        self.assertIn("a_output", out.columns)
        self.assertIn("b_output", out.columns)
        self.assertIn("combined", out.columns)

    def test_weighted_sum_math(self):
        # Default: w_1 = w_2 = 0.5; m1.scale = m2.scale = 1.0
        # output = 0.5 * x + 0.5 * x = x
        out = self.e.predict()
        x = self.data["x"].to_numpy()
        np.testing.assert_allclose(out["combined"].to_numpy(), x)

    def test_custom_weights(self):
        self.e.get_scalar_parameter("w_1").value = 0.3
        self.e.get_scalar_parameter("w_2").value = 0.7
        out = self.e.predict()
        x = self.data["x"].to_numpy()
        # output = 0.3 * 1 * x + 0.7 * 1 * x = x
        np.testing.assert_allclose(out["combined"].to_numpy(), x)

    def test_different_scales_propagate(self):
        # a_scale = 2.0, b_scale = 3.0; w_1 = w_2 = 0.5
        self.e.get_scalar_parameter("a_scale").value = 2.0
        self.e.get_scalar_parameter("b_scale").value = 3.0
        out = self.e.predict()
        x = self.data["x"].to_numpy()
        expected = 0.5 * (2.0 * x) + 0.5 * (3.0 * x)
        np.testing.assert_allclose(out["combined"].to_numpy(), expected)

    def test_state_predicted(self):
        self.e.predict()
        self.assertTrue(self.e.is_predicted())

    def test_finalize(self):
        self.e.predict()
        self.e.finalize()
        self.assertTrue(self.e.is_finalized())
        self.assertTrue(self.m1.is_finalized())
        self.assertTrue(self.m2.is_finalized())


class TestEnsembleProductMode(unittest.TestCase):

    def setUp(self):
        self.m1 = ScaleModel(input_col="x", output_col="y")
        self.m2 = ScaleModel(input_col="x", output_col="y")
        self.e = EnsembleModel(
            [(self.m1, lambda df: df["y"].to_numpy()),
             (self.m2, lambda df: df["y"].to_numpy())],
            mode="product",
            aliases=["a", "b"],
            output_name="combined",
        )
        self.e.initialize()
        self.e.validate()
        self.data = _make_data()
        self.e.prepare(self.data)

    def test_weight_defaults_product_mode(self):
        self.assertAlmostEqual(self.e.get_scalar_parameter("w_1").value, 1.0)
        self.assertAlmostEqual(self.e.get_scalar_parameter("w_2").value, 1.0)

    def test_product_math(self):
        # w_1 = w_2 = 1.0; scale = 1.0 → output = x^1 * x^1 = x^2
        out = self.e.predict()
        x = self.data["x"].to_numpy()
        np.testing.assert_allclose(out["combined"].to_numpy(), x ** 2)

    def test_no_normalize_constraint(self):
        self.assertNotIn("sum_w_leq_1", self.e.inequality_constraint_names)


class TestEnsembleParameterSync(unittest.TestCase):

    def test_sync_updates_child_values(self):
        m = ScaleModel()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], aliases=["m"])
        e.initialize()
        e.validate()
        data = _make_data()
        e.prepare(data)

        # Change prefixed param on ensemble
        e.get_scalar_parameter("m_scale").value = 5.0
        out = e.predict()

        # Child should receive the updated value
        x = data["x"].to_numpy()
        np.testing.assert_allclose(out["m_output"].to_numpy(), 5.0 * x)
        np.testing.assert_allclose(out["ensemble_output"].to_numpy(), 1.0 * 5.0 * x)

    def test_child_pre_initialized_accepted(self):
        m = ScaleModel()
        m.initialize()
        m.validate()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], aliases=["m"])
        e.initialize()
        self.assertIn("m_scale", e.scalar_parameter_names)


class TestEnsembleCalibrationProblemIntegration(unittest.TestCase):

    def test_calibration_problem_sees_all_params(self):
        try:
            from sparsehydro.calibration import CalibrationProblem, MSE
        except ImportError:
            self.skipTest("calibration extras not installed")

        data = _make_data()
        observed = data["x"].to_numpy() * 2.5  # target

        m1 = ScaleModel()
        m2 = ScaleModel()
        e = EnsembleModel(
            [(m1, lambda df: df["y"].to_numpy()),
             (m2, lambda df: df["y"].to_numpy())],
            aliases=["a", "b"],
        )
        e.initialize()
        e.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=e,
            data=data,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df: df["ensemble_output"].to_numpy(),
            },
        )
        # w_1, w_2, a_scale, b_scale = 4 calibratable params
        self.assertEqual(problem.n_params, 4)
        self.assertIn("a_scale", problem.param_names)
        self.assertIn("b_scale", problem.param_names)

    def test_calibration_problem_sees_constraints(self):
        try:
            from sparsehydro.calibration import CalibrationProblem, MSE
        except ImportError:
            self.skipTest("calibration extras not installed")

        data = _make_data()
        observed = data["x"].to_numpy()

        m = ConstrainedModel()
        e = EnsembleModel([(m, lambda df: df["y"].to_numpy())], aliases=["m"])
        e.initialize()
        e.validate()

        _obs = observed
        problem = CalibrationProblem(
            model=e,
            data=data,
            objectives=[MSE()],
            column_map={
                "observed":  lambda _: _obs,
                "predicted": lambda df: df["ensemble_output"].to_numpy(),
            },
        )
        # m_scale_leq_5 + sum_w_leq_1 = 2
        self.assertEqual(problem.n_ieq_constr, 2)
        self.assertIn("m_scale_leq_5", problem.constraint_names)
        self.assertIn("sum_w_leq_1", problem.constraint_names)


if __name__ == "__main__":
    unittest.main()
