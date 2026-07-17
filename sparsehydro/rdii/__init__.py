"""Backward-compatibility shim — re-exports from sparsehydro.models.rdii."""

from ..models.rdii import (
    IAModel,
    RTKTriangle,
    RTKEnsembleModel,
    triangular_uh,
    default_rtk_params,
    RDIIModel,
)

CombinedHydroModel = RDIIModel

__all__ = [
    "IAModel",
    "RTKTriangle",
    "RTKEnsembleModel",
    "triangular_uh",
    "default_rtk_params",
    "RDIIModel",
    "CombinedHydroModel",
]
