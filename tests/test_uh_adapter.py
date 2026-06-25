"""Tests for UnitHydrographAdapter and create_uh_model / register_all_uh_models.

UnitHydrograph (from uh_models.py) is not available in the test environment,
so it is fully mocked.  The tests verify adapter behaviour — lifecycle,
parameter sync, delegation calls — independently of the real UH implementation.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Build a minimal mock of the uh_models module so the lazy import resolves.
# ---------------------------------------------------------------------------

_MOCK_REGISTRY = {
    "Nash": {
        "func": MagicMock(),
        "pnames": ["A", "n", "k"],
        "p0":     [100.0, 2.0, 5.0],
        "lb":     [0.0,  0.01, 0.01],
        "ub":     [np.inf, 100.0, 500.0],
    },
    "Gamma": {
        "func": MagicMock(),
        "pnames": ["A", "tt", "tp"],
        "p0":     [1.0, 2.0, 5.0],
        "lb":     [0.0, 0.0, 0.0],
        "ub":     [np.inf, np.inf, np.inf],
    },
}


def _make_mock_uh_class() -> type:
    """Return a minimal UnitHydrograph mock class."""

    class MockUH:
        _registry = _MOCK_REGISTRY

        def __init__(self, model_name: str) -> None:
            if model_name not in self._registry:
                raise ValueError(f"Unknown model: {model_name}")
            self.model_name = model_name
            defn = self._registry[model_name]
            self.pnames = defn["pnames"]
            self.parameters = dict(zip(defn["pnames"], defn["p0"]))
            self.lb = defn["lb"]
            self.ub = defn["ub"]
            self.is_fit = False

        def predict(self, rain_stormflow, predict_range=None, trim=True):
            n = len(rain_stormflow)
            return pd.DataFrame({
                "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
                "Q_pred": np.ones(n) * self.parameters.get("A", 1.0),
            })

        def fit(self, rain_stormflow, **kwargs):
            # Simulate fitting by slightly adjusting A
            self.parameters["A"] = self.parameters.get("A", 1.0) * 1.1
            self.is_fit = True
            return dict(self.parameters)

        def get_uh(self, max_steps=864, norm=0):
            return np.ones(min(max_steps, 10))

    return MockUH


@pytest.fixture(autouse=True)
def mock_uh_module(monkeypatch):
    """Inject a mock uh_models module and reset the adapter's cached class."""
    mock_module = ModuleType("uh_models")
    mock_module.UnitHydrograph = _make_mock_uh_class()

    monkeypatch.setitem(sys.modules, "uh_models", mock_module)

    # Reset the lazy-import cache so each test gets a fresh mock class.
    import sparsehydro.models.unithydrograph.adapter as _adapter_mod
    monkeypatch.setattr(_adapter_mod, "_UnitHydrograph", None)

    yield mock_module.UnitHydrograph


@pytest.fixture()
def fresh_registry():
    """Isolated ModelRegistry for factory tests."""
    from sparsehydro.registry import ModelRegistry
    return ModelRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rain_df(n: int = 6) -> pd.DataFrame:
    return pd.DataFrame({
        "datetime": pd.date_range("2024-01-01", periods=n, freq="h"),
        "rain": np.linspace(5.0, 0.0, n),
        "stormflow": np.linspace(3.0, 0.0, n),
    })


# ---------------------------------------------------------------------------
# create_uh_model factory
# ---------------------------------------------------------------------------

class TestCreateUhModel:
    def test_returns_concrete_subclass(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import (
            UnitHydrographAdapter, create_uh_model,
        )
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            cls = create_uh_model("Nash")
        assert issubclass(cls, UnitHydrographAdapter)

    def test_model_name_slug(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            cls = create_uh_model("Nash")
        assert cls.model_name == "uh-nash"

    def test_uh_model_name_set(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            cls = create_uh_model("Gamma")
        assert cls._uh_model_name == "Gamma"

    def test_registered_in_registry(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            create_uh_model("Nash")
        assert fresh_registry.is_registered("uh-nash")

    def test_unknown_name_raises(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            with pytest.raises(KeyError, match="Ghost"):
                create_uh_model("Ghost")

    def test_duplicate_registration_raises(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            create_uh_model("Nash")
            with pytest.raises(ValueError, match="already registered"):
                create_uh_model("Nash")


# ---------------------------------------------------------------------------
# register_all_uh_models
# ---------------------------------------------------------------------------

class TestRegisterAll:
    def test_all_mock_models_registered(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import register_all_uh_models
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            result = register_all_uh_models()
        assert "uh-nash" in result
        assert "uh-gamma" in result
        assert len(result) == len(_MOCK_REGISTRY)

    def test_idempotent(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import register_all_uh_models
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            r1 = register_all_uh_models()
            r2 = register_all_uh_models()
        assert r1 == r2


# ---------------------------------------------------------------------------
# UnitHydrographAdapter lifecycle
# ---------------------------------------------------------------------------

class TestAdapterLifecycle:
    @pytest.fixture()
    def nash_cls(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            return create_uh_model("Nash")

    def test_initial_state(self, nash_cls):
        from sparsehydro.enums import ModelState
        m = nash_cls()
        assert m.state is ModelState.CREATED

    def test_initialize_registers_parameters(self, nash_cls):
        from sparsehydro.enums import ModelState
        m = nash_cls()
        m.initialize()
        assert m.is_initialized()
        assert set(m.scalar_parameter_names) == {"A", "n", "k"}

    def test_parameter_defaults(self, nash_cls):
        m = nash_cls()
        m.initialize()
        assert m.get_scalar_parameter("A").value == pytest.approx(100.0)
        assert m.get_scalar_parameter("n").value == pytest.approx(2.0)
        assert m.get_scalar_parameter("k").value == pytest.approx(5.0)

    def test_infinite_bounds_clamped(self, nash_cls):
        m = nash_cls()
        m.initialize()
        p_a = m.get_scalar_parameter("A")
        assert p_a.upper_bound < np.inf

    def test_validate_passes(self, nash_cls):
        m = nash_cls()
        m.initialize()
        assert m.validate() is True
        assert m.is_validated()

    def test_prepare_stores_data(self, nash_cls):
        m = nash_cls()
        m.initialize()
        m.validate()
        rain = _make_rain_df()
        m.prepare(rain)
        assert m.is_prepared()
        assert m._prepared_data is not None

    def test_predict_returns_dataframe(self, nash_cls):
        m = nash_cls()
        m.initialize()
        m.validate()
        m.prepare(_make_rain_df())
        result = m.predict()
        assert isinstance(result, pd.DataFrame)
        assert "Q_pred" in result.columns
        assert m.is_predicted()

    def test_finalize(self, nash_cls):
        m = nash_cls()
        m.initialize()
        m.validate()
        m.prepare(_make_rain_df())
        m.predict()
        m.finalize()
        assert m.is_finalized()
        assert m._prepared_data is None

    def test_predict_without_prepare_raises(self, nash_cls):
        m = nash_cls()
        m.initialize()
        with pytest.raises(RuntimeError, match="prepare"):
            m.predict()


# ---------------------------------------------------------------------------
# Parameter synchronisation
# ---------------------------------------------------------------------------

class TestParameterSync:
    @pytest.fixture()
    def nash_model(self, fresh_registry):
        from sparsehydro.models.unithydrograph.adapter import create_uh_model
        with patch("sparsehydro.models.unithydrograph.adapter.registry", fresh_registry):
            cls = create_uh_model("Nash")
        m = cls()
        m.initialize()
        return m

    def test_sync_to_uh_on_prepare(self, nash_model):
        nash_model.get_scalar_parameter("A").value = 200.0
        nash_model.prepare(_make_rain_df())
        assert nash_model._uh.parameters["A"] == pytest.approx(200.0)

    def test_sync_to_uh_on_predict(self, nash_model):
        nash_model.prepare(_make_rain_df())
        nash_model.get_scalar_parameter("A").value = 50.0
        nash_model.predict()
        assert nash_model._uh.parameters["A"] == pytest.approx(50.0)

    def test_sync_from_uh_after_fit(self, nash_model):
        original_a = nash_model.get_scalar_parameter("A").value
        nash_model.fit(_make_rain_df())
        fitted_a = nash_model.get_scalar_parameter("A").value
        # Mock fit multiplies A by 1.1
        assert fitted_a == pytest.approx(original_a * 1.1)

    def test_is_fit_reflects_uh_state(self, nash_model):
        assert not nash_model.is_fit
        nash_model.fit(_make_rain_df())
        assert nash_model.is_fit


# ---------------------------------------------------------------------------
# Base adapter guard
# ---------------------------------------------------------------------------

class TestBaseAdapterGuard:
    def test_direct_instantiation_raises_on_initialize(self):
        from sparsehydro.models.unithydrograph.adapter import UnitHydrographAdapter
        m = UnitHydrographAdapter()
        with pytest.raises(TypeError, match="_uh_model_name"):
            m.initialize()
