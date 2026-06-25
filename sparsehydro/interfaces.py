"""Backward-compatibility shim — re-exports from sparsehydro.models.base."""

from .models.base import IModel, IUnitHydroComponent

__all__ = ["IModel", "IUnitHydroComponent"]
