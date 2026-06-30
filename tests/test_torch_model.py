"""Tests for ITorchModel (skipped when PyTorch is not installed)."""

import pytest

torch = pytest.importorskip("torch", reason="PyTorch not installed")
import torch.nn as nn
import pandas as pd

from sparsehydro.enums import ModelState
from sparsehydro.parameters import ScalarParameter
from sparsehydro.models.torch_model import ITorchModel


class LinearReservoir(ITorchModel):
    """Minimal differentiable reservoir model for testing."""

    model_name = "linear-reservoir-torch"

    def initialize(self) -> None:
        self.k = nn.Parameter(torch.tensor(0.5))
        self.register_scalar_parameter(
            ScalarParameter("k", value=0.5, lower_bound=0.0, upper_bound=1.0,
                            description="recession coefficient")
        )
        self._state = ModelState.INITIALIZED

    def validate(self) -> bool:
        ok = self.parameters_valid()
        if ok:
            self._state = ModelState.VALIDATED
        return ok

    def prepare(self, forcing: "torch.Tensor") -> None:
        self._forcing = forcing
        self._state = ModelState.PREPARED

    def forward(self, forcing: "torch.Tensor") -> "torch.Tensor":
        return self.k * forcing

    def finalize(self) -> None:
        self._state = ModelState.FINALIZED


class TestITorchModel:
    def setup_method(self):
        self.model = LinearReservoir()

    def test_is_nn_module(self):
        assert isinstance(self.model, nn.Module)

    def test_initial_state(self):
        assert self.model.state is ModelState.CREATED

    def test_initialize(self):
        self.model.initialize()
        assert self.model.is_initialized()
        assert "k" in self.model.scalar_parameter_names

    def test_validate(self):
        self.model.initialize()
        assert self.model.validate() is True
        assert self.model.is_validated()

    def test_forward_and_predict(self):
        self.model.initialize()
        self.model.validate()
        forcing = torch.tensor([1.0, 2.0, 3.0])
        self.model.prepare(forcing)
        result = self.model.predict(forcing)
        assert isinstance(result, torch.Tensor)
        assert result.shape == (3,)
        assert self.model.is_predicted()

    def test_gradients_flow(self):
        self.model.initialize()
        forcing = torch.tensor([1.0, 2.0, 3.0])
        out = self.model.forward(forcing)
        loss = out.sum()
        loss.backward()
        assert self.model.k.grad is not None

    def test_finalize(self):
        self.model.initialize()
        self.model.validate()
        forcing = torch.tensor([1.0])
        self.model.prepare(forcing)
        self.model.predict(forcing)
        self.model.finalize()
        assert self.model.is_finalized()

    def test_nn_parameters_discoverable(self):
        self.model.initialize()
        params = list(self.model.parameters())
        assert len(params) == 1
