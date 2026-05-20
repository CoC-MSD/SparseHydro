"""sparsehydro.calibration.solvers — solver abstractions and implementations."""

from .base import ISolver
from .nsga2 import NSGAIISolver
from .scipy_solver import ScipySolver

__all__ = ["ISolver", "NSGAIISolver", "ScipySolver"]
