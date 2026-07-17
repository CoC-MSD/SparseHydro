"""Tests for ModelRegistry."""

import pytest

from sparsehydro.enums import ModelState
from sparsehydro.models import IModel
from sparsehydro.parameters import ScalarParameter
from sparsehydro.registry import ModelRegistry


# ---------------------------------------------------------------------------
# Helper fixtures — isolated registry per test to avoid cross-test pollution
# ---------------------------------------------------------------------------

@pytest.fixture()
def reg() -> ModelRegistry:
    return ModelRegistry()


def _make_concrete(name: str) -> type[IModel]:
    """Dynamically create a minimal concrete IModel with the given model_name."""

    def initialize(self):
        self._state = ModelState.INITIALIZED

    def validate(self):
        self._state = ModelState.VALIDATED
        return True

    def prepare(self, *a, **kw):
        self._state = ModelState.PREPARED

    def predict(self, *a, **kw):
        import pandas as pd
        self._state = ModelState.PREDICTED
        return pd.Series(dtype=float)

    def finalize(self):
        self._state = ModelState.FINALIZED

    return type(
        f"Model_{name}",
        (IModel,),
        {
            "model_name": name,
            "initialize": initialize,
            "validate": validate,
            "prepare": prepare,
            "predict": predict,
            "finalize": finalize,
        },
    )


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------

class TestRegister:
    def test_register_returns_class(self, reg):
        cls = _make_concrete("alpha")
        assert reg.register(cls) is cls

    def test_decorator_usage(self, reg):
        @reg.register
        class MyModel(IModel):
            model_name = "decorator-model"

            def initialize(self): self._state = ModelState.INITIALIZED
            def validate(self): self._state = ModelState.VALIDATED; return True
            def prepare(self, *a, **kw): self._state = ModelState.PREPARED
            def predict(self, *a, **kw):
                import pandas as pd; self._state = ModelState.PREDICTED; return pd.Series(dtype=float)
            def finalize(self): self._state = ModelState.FINALIZED

        assert reg.is_registered("decorator-model")

    def test_register_duplicate_raises(self, reg):
        cls = _make_concrete("dup")
        reg.register(cls)
        with pytest.raises(ValueError, match="already registered"):
            reg.register(cls)

    def test_register_non_imodel_raises(self, reg):
        with pytest.raises(TypeError, match="not a subclass of IModel"):
            reg.register(object)  # type: ignore[arg-type]

    def test_register_abstract_raises(self, reg):
        with pytest.raises(TypeError, match="abstract"):
            reg.register(IModel)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# model_name enforcement on IModel subclasses
# ---------------------------------------------------------------------------

class TestModelNameEnforcement:
    def test_concrete_without_model_name_raises(self):
        with pytest.raises(TypeError, match="model_name"):
            type(
                "NoName",
                (IModel,),
                {
                    "initialize": lambda self: None,
                    "validate": lambda self: True,
                    "prepare": lambda self: None,
                    "predict": lambda self: None,
                    "finalize": lambda self: None,
                },
            )

    def test_empty_model_name_raises(self):
        with pytest.raises(TypeError, match="non-empty"):
            type(
                "EmptyName",
                (IModel,),
                {
                    "model_name": "",
                    "initialize": lambda self: None,
                    "validate": lambda self: True,
                    "prepare": lambda self: None,
                    "predict": lambda self: None,
                    "finalize": lambda self: None,
                },
            )

    def test_abstract_subclass_without_model_name_is_allowed(self):
        from abc import abstractmethod

        class AbstractIntermediate(IModel):
            @abstractmethod
            def extra(self) -> None: ...

        # should not raise — still abstract
        assert AbstractIntermediate.__abstractmethods__


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------

class TestGet:
    def test_get_registered_class(self, reg):
        cls = _make_concrete("beta")
        reg.register(cls)
        assert reg.get("beta") is cls

    def test_get_missing_raises(self, reg):
        with pytest.raises(KeyError, match="missing"):
            reg.get("missing")

    def test_create_returns_instance(self, reg):
        cls = _make_concrete("gamma")
        reg.register(cls)
        instance = reg.create("gamma")
        assert isinstance(instance, IModel)
        assert instance.state is ModelState.CREATED

    def test_create_missing_raises(self, reg):
        with pytest.raises(KeyError):
            reg.create("ghost")


# ---------------------------------------------------------------------------
# Unregister
# ---------------------------------------------------------------------------

class TestUnregister:
    def test_unregister_removes_model(self, reg):
        cls = _make_concrete("delta")
        reg.register(cls)
        reg.unregister("delta")
        assert not reg.is_registered("delta")

    def test_unregister_missing_raises(self, reg):
        with pytest.raises(KeyError, match="epsilon"):
            reg.unregister("epsilon")

    def test_re_register_after_unregister(self, reg):
        cls = _make_concrete("zeta")
        reg.register(cls)
        reg.unregister("zeta")
        reg.register(cls)
        assert reg.is_registered("zeta")


# ---------------------------------------------------------------------------
# Collections protocol
# ---------------------------------------------------------------------------

class TestCollections:
    def test_len(self, reg):
        assert len(reg) == 0
        reg.register(_make_concrete("a"))
        reg.register(_make_concrete("b"))
        assert len(reg) == 2

    def test_contains(self, reg):
        reg.register(_make_concrete("eta"))
        assert "eta" in reg
        assert "theta" not in reg

    def test_iter_sorted(self, reg):
        reg.register(_make_concrete("c"))
        reg.register(_make_concrete("a"))
        reg.register(_make_concrete("b"))
        assert list(reg) == ["a", "b", "c"]

    def test_names_sorted(self, reg):
        reg.register(_make_concrete("z"))
        reg.register(_make_concrete("m"))
        reg.register(_make_concrete("a"))
        assert reg.names() == ["a", "m", "z"]

    def test_is_registered(self, reg):
        assert not reg.is_registered("iota")
        reg.register(_make_concrete("iota"))
        assert reg.is_registered("iota")
