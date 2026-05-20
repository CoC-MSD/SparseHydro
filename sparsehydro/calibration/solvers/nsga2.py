"""NSGA-II multi-objective solver backed by pymoo.

Requires the optional ``pymoo`` dependency::

    pip install sparsehydro[rdii]

The solver wraps any :class:`~sparsehydro.calibration.problem.CalibrationProblem`
inside a ``pymoo.ElementwiseProblem`` adapter and runs NSGA-II with SBX
crossover and polynomial mutation.  All objectives are passed to NSGA-II in
minimisation form (maximised objectives are negated by
:meth:`~sparsehydro.calibration.problem.CalibrationProblem.evaluate`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord, _identify_pareto

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


try:
    from pymoo.algorithms.moo.nsga2 import NSGA2  # type: ignore[import]
    from pymoo.core.callback import Callback  # type: ignore[import]
    from pymoo.core.problem import ElementwiseProblem  # type: ignore[import]
    from pymoo.operators.crossover.sbx import SBX  # type: ignore[import]
    from pymoo.operators.mutation.pm import PM  # type: ignore[import]
    from pymoo.operators.sampling.rnd import FloatRandomSampling  # type: ignore[import]
    from pymoo.optimize import minimize as pymoo_minimize  # type: ignore[import]

    class _PymooAdapter(ElementwiseProblem):
        """pymoo ElementwiseProblem backed by a CalibrationProblem."""

        def __init__(self, problem: "CalibrationProblem") -> None:
            xl, xu = problem.bounds
            super().__init__(
                n_var=problem.n_params,
                n_obj=problem.n_objectives,
                xl=xl,
                xu=xu,
            )
            self._problem = problem

        def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
            out["F"] = self._problem.evaluate(x)

    class _GenerationCallback(Callback):
        """Appends a GenerationRecord to history after each generation."""

        def __init__(self) -> None:
            super().__init__()
            self.history: list[GenerationRecord] = []

        def notify(self, algorithm) -> None:  # type: ignore[override]
            pop = algorithm.pop
            X = pop.get("X").copy()
            F = pop.get("F").copy()
            try:
                ranks = pop.get("rank")
                n_pareto = int(np.sum(ranks == 0))
            except Exception:
                n_pareto = int(np.sum(_identify_pareto(F)))
            self.history.append(
                GenerationRecord(
                    generation=int(algorithm.n_gen),
                    X=X,
                    F=F,
                    n_pareto=n_pareto,
                )
            )

    class NSGAIISolver(ISolver):
        """Multi-objective NSGA-II calibration solver.

        Works with any :class:`~sparsehydro.calibration.problem.CalibrationProblem`.
        For multi-objective problems the full Pareto front is returned in
        :attr:`~sparsehydro.calibration.result.CalibrationResult.pareto_X` and
        :attr:`~sparsehydro.calibration.result.CalibrationResult.pareto_F`.

        :param pop_size: Population size per generation.
        :param n_gen: Total number of generations.
        :param seed: Random seed for reproducibility (``None`` for random).
        :param verbose: Print pymoo progress output.
        :param n_jobs: Parallel workers via ``ProcessPoolExecutor``
            (``1`` = sequential).
        """

        def __init__(
            self,
            pop_size: int = 100,
            n_gen: int = 200,
            seed: int | None = 42,
            verbose: bool = False,
            n_jobs: int = 1,
        ) -> None:
            self.pop_size = pop_size
            self.n_gen = n_gen
            self.seed = seed
            self.verbose = verbose
            self.n_jobs = n_jobs

        def solve(self, problem: "CalibrationProblem") -> CalibrationResult:
            """Run NSGA-II and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem in PREPARED state.
            :type problem: CalibrationProblem
            :returns: Result with per-generation history and final Pareto front.
            :rtype: CalibrationResult
            """
            worker_problem = problem.make_copy()
            pymoo_problem = _PymooAdapter(worker_problem)

            if self.n_jobs > 1:
                try:
                    from concurrent.futures import ProcessPoolExecutor
                    from pymoo.core.problem import (  # type: ignore[import]
                        StarmapParallelization,
                    )
                    executor = ProcessPoolExecutor(max_workers=self.n_jobs)
                    pymoo_problem.runner = StarmapParallelization(executor.map)
                except Exception:
                    pass  # fall back to sequential

            callback = _GenerationCallback()
            algorithm = NSGA2(
                pop_size=self.pop_size,
                sampling=FloatRandomSampling(),
                crossover=SBX(prob=0.9, eta=15),
                mutation=PM(eta=20),
                eliminate_duplicates=True,
            )
            res = pymoo_minimize(
                pymoo_problem,
                algorithm,
                ("n_gen", self.n_gen),
                callback=callback,
                seed=self.seed,
                verbose=self.verbose,
            )
            return CalibrationResult(
                history=callback.history,
                pareto_X=res.X,
                pareto_F=res.F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

except ImportError:

    class _NSGAIISolverStub:
        """Placeholder — requires ``pymoo``.

        Install with::

            pip install sparsehydro[rdii]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "pymoo is required for NSGAIISolver. "
                "Install with: pip install sparsehydro[rdii]"
            )

        def solve(self, problem) -> None:
            raise ImportError(
                "pymoo is required for NSGAIISolver. "
                "Install with: pip install sparsehydro[rdii]"
            )

    NSGAIISolver = _NSGAIISolverStub  # type: ignore[assignment, misc]
