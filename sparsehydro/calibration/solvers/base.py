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

    ``solve()`` accepts ``**kwargs`` that override constructor-level settings for
    that call only — the solver instance is not mutated.  This lets the same solver
    be reused for both quick exploratory runs and full production runs::

        solver = NSGAIISolver(pop_size=100, n_gen=200)
        quick  = solver.solve(problem, n_gen=5)    # fast sanity check
        full   = solver.solve(problem)             # original settings

    Minimal implementation::

        class MySolver(ISolver):
            def solve(self, problem: CalibrationProblem, **kwargs) -> CalibrationResult:
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
    def solve(self, problem: "CalibrationProblem", **kwargs) -> CalibrationResult:
        """Run the solver and return results.

        :param problem: Calibration problem wrapping model + data + objectives.
        :param kwargs: Per-call overrides for solver settings (e.g. ``n_gen=10``).
            Each solver documents which keys are recognised.
        :returns: Calibration result with Pareto front and generation history.
        :rtype: CalibrationResult
        """
