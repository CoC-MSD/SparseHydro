"""Particle Swarm Optimization solver backed by Platypus.

Uses :class:`platypus.SMPSO` (Speed-constrained Multi-objective PSO) by default.
Passing ``epsilons`` switches to :class:`platypus.OMOPSO` (which maintains an
ε-dominance archive instead of a standard Pareto archive).

Requires the optional ``platypus`` dependency::

    pip install sparsehydro[platypus]

Example usage::

    from sparsehydro.calibration import ParticleSwarmSolver

    # SMPSO (default)
    solver = ParticleSwarmSolver(swarm_size=50, n_evaluations=5_000)

    # OMOPSO with per-objective epsilon archive
    solver = ParticleSwarmSolver(
        swarm_size=50,
        n_evaluations=5_000,
        epsilons=[0.01, 0.01],
    )

    result = solver.solve(problem)
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord, _identify_pareto

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


try:
    import platypus  # type: ignore[import]

    class ParticleSwarmSolver(ISolver):
        """Multi-objective Particle Swarm solver.

        Wraps :class:`platypus.SMPSO` (default) or :class:`platypus.OMOPSO`
        in the :class:`~sparsehydro.calibration.solvers.ISolver` interface.

        Both algorithms maintain a Pareto *leader archive* that guides particle
        velocities; this archive is used as the Pareto front in the returned
        :class:`~sparsehydro.calibration.result.CalibrationResult` and is also
        recorded in :attr:`~sparsehydro.calibration.result.CalibrationResult.history`
        at every ``record_frequency`` steps.

        :param swarm_size: Number of particles.
        :type swarm_size: int
        :param leader_size: Maximum size of the Pareto leader archive.
        :type leader_size: int
        :param n_evaluations: Total function-evaluation budget.  The solver
            stops when ``algorithm.nfe >= n_evaluations``.
        :type n_evaluations: int
        :param seed: Seed for :mod:`random` and :mod:`numpy.random` (``None``
            for non-deterministic runs).
        :type seed: int | None
        :param record_frequency: Append a
            :class:`~sparsehydro.calibration.result.GenerationRecord` to
            history every *record_frequency* steps.
        :type record_frequency: int
        :param epsilons: Per-objective ε values for the ε-dominance archive.
            When provided, :class:`platypus.OMOPSO` is used instead of
            :class:`platypus.SMPSO`.  Must contain one value per objective.
        :type epsilons: list[float] | None

        Examples::

            from sparsehydro.calibration import ParticleSwarmSolver

            # SMPSO — no extra configuration needed
            solver = ParticleSwarmSolver(swarm_size=50, n_evaluations=5_000)

            # OMOPSO — supply one epsilon per objective
            solver = ParticleSwarmSolver(
                swarm_size=50,
                n_evaluations=5_000,
                epsilons=[0.01, 0.01],   # two objectives
            )

            result = solver.solve(problem)
        """

        def __init__(
            self,
            swarm_size: int = 100,
            leader_size: int = 100,
            n_evaluations: int = 10_000,
            seed: int | None = 42,
            record_frequency: int = 1,
            epsilons: list[float] | None = None,
        ) -> None:
            self.swarm_size = swarm_size
            self.leader_size = leader_size
            self.n_evaluations = n_evaluations
            self.seed = seed
            self.record_frequency = record_frequency
            self.epsilons = epsilons

        def solve(self, problem: "CalibrationProblem") -> CalibrationResult:
            """Run the PSO algorithm and return a
            :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem in PREPARED state.
            :type problem: CalibrationProblem
            :returns: Result with per-step leader-archive history and final
                Pareto front.
            :rtype: CalibrationResult
            """
            if self.seed is not None:
                _random.seed(self.seed)
                np.random.seed(self.seed)

            worker = problem.make_copy()
            xl, xu = worker.bounds
            n_params = worker.n_params
            n_obj = worker.n_objectives

            # Validate epsilons length against objective count
            if self.epsilons is not None and len(self.epsilons) != n_obj:
                raise ValueError(
                    f"epsilons length ({len(self.epsilons)}) must equal the number "
                    f"of objectives ({n_obj})."
                )

            # Build the platypus Problem ------------------------------------------
            platypus_problem = platypus.Problem(n_params, n_obj)
            platypus_problem.types[:] = [
                platypus.Real(float(lo), float(hi)) for lo, hi in zip(xl, xu)
            ]
            platypus_problem.directions[:] = [platypus.Problem.MINIMIZE] * n_obj

            def _evaluate(vars_list: list) -> list:
                x = np.array(vars_list, dtype=float)
                return worker.evaluate(x).tolist()

            platypus_problem.function = _evaluate

            # Instantiate SMPSO or OMOPSO ----------------------------------------
            if self.epsilons is not None:
                algorithm = platypus.OMOPSO(
                    platypus_problem,
                    epsilons=list(self.epsilons),
                    swarm_size=self.swarm_size,
                    leader_size=self.leader_size,
                )
            else:
                algorithm = platypus.SMPSO(
                    platypus_problem,
                    swarm_size=self.swarm_size,
                    leader_size=self.leader_size,
                )

            # Run via step() so algorithm.result is updated after every step -----
            history: list[GenerationRecord] = []
            step_count = 0

            while algorithm.nfe < self.n_evaluations:
                algorithm.step()
                step_count += 1

                if step_count % self.record_frequency == 0:
                    leaders = algorithm.result
                    if leaders:
                        X = np.array([list(sol.variables) for sol in leaders], dtype=float)
                        F = np.array([list(sol.objectives) for sol in leaders], dtype=float)
                        pareto_mask = _identify_pareto(F)
                        history.append(
                            GenerationRecord(
                                generation=step_count,
                                X=X,
                                F=F,
                                n_pareto=int(np.sum(pareto_mask)),
                            )
                        )

            # Final Pareto front --------------------------------------------------
            final_leaders = algorithm.result
            if final_leaders:
                all_X = np.array([list(sol.variables) for sol in final_leaders], dtype=float)
                all_F = np.array([list(sol.objectives) for sol in final_leaders], dtype=float)
                pareto_mask = _identify_pareto(all_F)
                pareto_X = all_X[pareto_mask]
                pareto_F = all_F[pareto_mask]
            else:
                pareto_X = np.empty((0, n_params), dtype=float)
                pareto_F = np.empty((0, n_obj), dtype=float)

            return CalibrationResult(
                history=history,
                pareto_X=pareto_X,
                pareto_F=pareto_F,
                param_names=problem.param_names,
                objective_names=problem.objective_names,
                minimize_flags=problem.minimize_flags,
            )

except ImportError:

    class _ParticleSwarmSolverStub:
        """Placeholder — requires ``platypus-opt``.

        Install with::

            pip install sparsehydro[platypus]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "platypus-opt is required for ParticleSwarmSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

        def solve(self, problem) -> None:  # type: ignore[return]
            raise ImportError(
                "platypus-opt is required for ParticleSwarmSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

    ParticleSwarmSolver = _ParticleSwarmSolverStub  # type: ignore[assignment, misc]
