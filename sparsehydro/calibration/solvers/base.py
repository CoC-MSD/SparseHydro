"""Abstract solver interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from ..result import CalibrationResult

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


class ISolver(ABC):
    """Abstract calibration solver.

    All solvers accept a :class:`~sparsehydro.calibration.problem.CalibrationProblem`
    and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

    Minimal implementation::

        class MySolver(ISolver):
            def solve(self, problem: CalibrationProblem) -> CalibrationResult:
                x_best = ...          # your optimisation logic
                F_best = problem.evaluate(x_best).reshape(1, -1)
                return CalibrationResult(
                    history=[],
                    pareto_X=x_best.reshape(1, -1),
                    pareto_F=F_best,
                    param_names=problem.param_names,
                    objective_names=problem.objective_names,
                    minimize_flags=problem.minimize_flags,
                )
    """

    @abstractmethod
    def solve(self, problem: "CalibrationProblem") -> CalibrationResult:
        """Run the solver and return results.

        :param problem: Calibration problem wrapping model + objectives + data.
        :type problem: CalibrationProblem
        :returns: Calibration result with Pareto front and generation history.
        :rtype: CalibrationResult
        """
