"""sparsehydro — interfaces and utilities for parsimonious hydrological models."""

from .enums import ModelState
from .interfaces import IModel
from .parameters import ScalarParameter, VectorParameter
from .registry import ModelRegistry, registry
from .unithydrograph import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
)

__version__ = "0.1.0"
__all__ = [
    "ModelState",
    "ScalarParameter",
    "VectorParameter",
    "IModel",
    "ModelRegistry",
    "registry",
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
    "ITorchModel",
    "__version__",
]

try:
    from .torch_model import ITorchModel
except ImportError:  # pragma: no cover
    pass
