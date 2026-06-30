"""Tests for the IModel abstract interface."""

import pandas as pd
import pytest

from sparsehydro.enums import ModelState
from sparsehydro.models import IModel
from sparsehydro.parameters import ScalarParameter, VectorParameter


class SimpleModel(IModel):
    """Minimal concrete implementation of IModel for testing."""

    model_name = "simple-model"

    def initialize(self) -> None:
        self.register_scalar_parameter(
            ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0)
        )
        self.register_vector_parameter(
            VectorParameter("beta", values=[1.0, 2.0], lower_bounds=0.0, upper_bounds=5.0)
        )
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, data=None) -> None:
        self._data = data
        self._state = ModelState.PREPARED

    def predict(self) -> pd.Series:
        self._state = ModelState.PREDICTED
        return pd.Series([1.0, 2.0, 3.0], name="Q")

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


class TestModelState:
    def test_initial_state(self):
        m = SimpleModel()
        assert m.state is ModelState.CREATED
        assert m.is_created()
        assert not m.is_initialized()

    def test_full_lifecycle(self):
        m = SimpleModel()

        m.initialize()
        assert m.is_initialized()

        assert m.validate() is True
        assert m.is_validated()

        m.prepare(data=[1, 2, 3])
        assert m.is_prepared()

        result = m.predict()
        assert isinstance(result, (pd.DataFrame, pd.Series))
        assert m.is_predicted()

        m.finalize()
        assert m.is_finalized()

    def test_state_flags_are_exclusive(self):
        m = SimpleModel()
        m.initialize()
        assert not m.is_created()
        assert m.is_initialized()
        assert not m.is_validated()
        assert not m.is_prepared()
        assert not m.is_predicted()
        assert not m.is_finalized()


class TestParameterRegistry:
    def setup_method(self):
        self.model = SimpleModel()
        self.model.initialize()

    def test_scalar_parameter_names(self):
        assert "k" in self.model.scalar_parameter_names

    def test_vector_parameter_names(self):
        assert "beta" in self.model.vector_parameter_names

    def test_get_scalar_parameter(self):
        p = self.model.get_scalar_parameter("k")
        assert p.name == "k"
        assert p.value == pytest.approx(0.5)

    def test_get_vector_parameter(self):
        p = self.model.get_vector_parameter("beta")
        assert p.name == "beta"
        assert p.size == 2

    def test_get_missing_scalar_raises(self):
        with pytest.raises(KeyError, match="missing"):
            self.model.get_scalar_parameter("missing")

    def test_get_missing_vector_raises(self):
        with pytest.raises(KeyError, match="missing"):
            self.model.get_vector_parameter("missing")

    def test_parameters_valid_true(self):
        assert self.model.parameters_valid()

    def test_parameters_valid_false_after_out_of_bounds(self):
        self.model._scalar_parameters["k"].value = 99.0
        assert not self.model.parameters_valid()

    def test_validate_returns_false_when_invalid(self):
        self.model._scalar_parameters["k"].value = 99.0
        assert self.model.validate() is False
        assert not self.model.is_validated()

    def test_register_overwrites_existing(self):
        new_p = ScalarParameter("k", value=0.9, lower_bound=0.0, upper_bound=1.0)
        self.model.register_scalar_parameter(new_p)
        assert self.model.get_scalar_parameter("k").value == pytest.approx(0.9)


class TestRenameParameter:
    def setup_method(self):
        self.model = SimpleModel()
        self.model.initialize()

    def test_rename_scalar_updates_registry_key(self):
        self.model.rename_scalar_parameter("k", "recession_rate")
        assert "recession_rate" in self.model.scalar_parameter_names
        assert "k" not in self.model.scalar_parameter_names

    def test_rename_scalar_updates_param_name_field(self):
        self.model.rename_scalar_parameter("k", "recession_rate")
        p = self.model.get_scalar_parameter("recession_rate")
        assert p.name == "recession_rate"

    def test_rename_scalar_preserves_value(self):
        self.model.rename_scalar_parameter("k", "recession_rate")
        p = self.model.get_scalar_parameter("recession_rate")
        assert p.value == pytest.approx(0.5)

    def test_rename_scalar_missing_raises(self):
        with pytest.raises(KeyError):
            self.model.rename_scalar_parameter("nonexistent", "new_name")

    def test_rename_scalar_duplicate_raises(self):
        self.model.register_scalar_parameter(
            ScalarParameter("other", value=0.1, lower_bound=0.0, upper_bound=1.0)
        )
        with pytest.raises(ValueError, match="already exists"):
            self.model.rename_scalar_parameter("k", "other")

    def test_rename_vector_updates_registry_key(self):
        self.model.rename_vector_parameter("beta", "shape_params")
        assert "shape_params" in self.model.vector_parameter_names
        assert "beta" not in self.model.vector_parameter_names

    def test_rename_vector_updates_param_name_field(self):
        self.model.rename_vector_parameter("beta", "shape_params")
        p = self.model.get_vector_parameter("shape_params")
        assert p.name == "shape_params"

    def test_rename_vector_missing_raises(self):
        with pytest.raises(KeyError):
            self.model.rename_vector_parameter("nonexistent", "new_name")

    def test_rename_vector_duplicate_raises(self):
        with pytest.raises(ValueError, match="already exists"):
            self.model.rename_vector_parameter("beta", "beta")
