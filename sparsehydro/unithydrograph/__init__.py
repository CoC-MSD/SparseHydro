"""Backward-compatibility shim — re-exports from sparsehydro.models.unithydrograph."""

from ..models.unithydrograph import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
    GammaUH,
    NashUH,
    TriangleUH,
    SequentialFitter,
    SequentialFitSummary,
)

__all__ = [
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
    "GammaUH",
    "NashUH",
    "TriangleUH",
    "SequentialFitter",
    "SequentialFitSummary",
]
