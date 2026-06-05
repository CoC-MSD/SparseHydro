"""Generic Platypus multi-objective solver.

Wraps any :class:`platypus.Algorithm` subclass so that the full Platypus suite —
NSGA-II, NSGA-III, SPEA2, MOEA/D, GDE3, IBEA, OMOPSO, SMPSO, ε-MOEA, and more —
can calibrate any :class:`~sparsehydro.calibration.problem.CalibrationProblem`.

Requires the optional ``platypus`` dependency::

    pip install sparsehydro[platypus]

Example usage::

    import platypus
    from sparsehydro.calibration import PlatypusSolver

    # Any Platypus algorithm, any keyword arguments
    solver = PlatypusSolver(platypus.NSGAII, population_size=100, n_evaluations=10_000)
    solver = PlatypusSolver(platypus.SPEA2, population_size=50, n_evaluations=5_000)
    solver = PlatypusSolver(platypus.GDE3, population_size=80, n_evaluations=8_000)
    solver = PlatypusSolver(platypus.IBEA, population_size=100, n_evaluations=10_000)
"""

from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING, Type

import numpy as np

from .base import ISolver
from ..result import CalibrationResult, GenerationRecord, _identify_pareto

if TYPE_CHECKING:
    from ..problem import CalibrationProblem


try:
    import platypus  # type: ignore[import]

    class PlatypusSolver(ISolver):
        """Generic Platypus multi-objective solver.

        Wraps any :class:`platypus.Algorithm` subclass — NSGA-II, NSGA-III,
        SPEA2, MOEA/D, GDE3, IBEA, OMOPSO, SMPSO, ε-MOEA, etc. — in the
        :class:`~sparsehydro.calibration.solvers.ISolver` interface.

        The algorithm class and all of its constructor keyword arguments (except
        ``problem``, which is built internally) are passed directly through, so
        any algorithm-specific tuning is fully accessible.

        :param algorithm_class: A Platypus algorithm *class* (not instance), e.g.
            ``platypus.NSGA2``, ``platypus.SPEA2``, ``platypus.MOEAD``.
        :type algorithm_class: type
        :param n_evaluations: Total function-evaluation budget.  The solver stops
            when ``algorithm.nfe >= n_evaluations``.
        :type n_evaluations: int
        :param seed: Seed for :mod:`random` and :mod:`numpy.random` (``None``
            for non-deterministic runs).
        :type seed: int | None
        :param record_frequency: Append a :class:`~sparsehydro.calibration.result.GenerationRecord`
            to history every *record_frequency* algorithm iterations.  Set to a
            larger value to reduce memory usage on long runs.
        :type record_frequency: int
        :param algorithm_kwargs: All remaining keyword arguments are forwarded
            verbatim to ``algorithm_class(problem, **algorithm_kwargs)``.

        Examples::

            import platypus
            from sparsehydro.calibration import PlatypusSolver

            # NSGA-II with explicit population size
            solver = PlatypusSolver(platypus.NSGAII, population_size=100, n_evaluations=10_000)

            # SPEA2
            solver = PlatypusSolver(platypus.SPEA2, population_size=50, n_evaluations=5_000)

            # GDE3 (good for real-valued problems)
            solver = PlatypusSolver(platypus.GDE3, population_size=80, n_evaluations=8_000)

            # ε-MOEA with epsilon archive
            solver = PlatypusSolver(
                platypus.EpsMOEA,
                epsilons=[0.01, 0.01],
                population_size=100,
                n_evaluations=10_000,
            )

            result = solver.solve(problem)
        """

        def __init__(
            self,
            algorithm_class: Type[platypus.Algorithm],
            n_evaluations: int = 10_000,
            seed: int | None = 42,
            record_frequency: int = 1,
            **algorithm_kwargs,
        ) -> None:
            self.algorithm_class = algorithm_class
            self.n_evaluations = n_evaluations
            self.seed = seed
            self.record_frequency = record_frequency
            self.algorithm_kwargs = algorithm_kwargs

        def solve(self, problem: "CalibrationProblem", **kwargs) -> CalibrationResult:
            """Run the Platypus algorithm and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem wrapping model + data + objectives.
            :param kwargs: Per-call overrides: ``n_evaluations``, ``seed``,
                ``record_frequency``.  Additional keys are merged into
                ``algorithm_kwargs`` and forwarded to the algorithm constructor.
            :returns: Result with per-iteration history and final Pareto front.
            :rtype: CalibrationResult
            """
            n_evaluations    = kwargs.pop("n_evaluations",    self.n_evaluations)
            seed             = kwargs.pop("seed",             self.seed)
            record_frequency = kwargs.pop("record_frequency", self.record_frequency)
            algorithm_kwargs = {**self.algorithm_kwargs, **kwargs}

            if seed is not None:
                _random.seed(seed)
                np.random.seed(seed)

            worker = problem.make_copy()
            xl, xu = worker.bounds
            n_params = worker.n_params
            n_obj = worker.n_objectives

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

            # Build and step the algorithm ----------------------------------------
            algorithm = self.algorithm_class(platypus_problem, **algorithm_kwargs)

            history: list[GenerationRecord] = []
            iteration = 0

            # Use step() rather than initialize()/iterate() directly so that
            # algorithm.result is updated after each step (it is only written
            # inside step(), not inside the lower-level methods).
            while algorithm.nfe < n_evaluations:
                algorithm.step()
                iteration += 1

                if iteration % record_frequency == 0:
                    current = algorithm.result
                    if current:
                        X = np.array(
                            [list(sol.variables) for sol in current], dtype=float
                        )
                        F = np.array(
                            [list(sol.objectives) for sol in current], dtype=float
                        )
                        pareto_mask = _identify_pareto(F)
                        history.append(
                            GenerationRecord(
                                generation=iteration,
                                X=X,
                                F=F,
                                n_pareto=int(np.sum(pareto_mask)),
                            )
                        )

            # Final Pareto front --------------------------------------------------
            final = algorithm.result
            if final:
                all_X = np.array([list(sol.variables) for sol in final], dtype=float)
                all_F = np.array([list(sol.objectives) for sol in final], dtype=float)
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

    class ParticleSwarmSolver(ISolver):
        """Multi-objective Particle Swarm solver backed by Platypus.

        Selects ``platypus.OMOPSO`` (ε-dominance archive) when *epsilons* are
        provided, otherwise uses ``platypus.SMPSO``.

        :param swarm_size: Number of particles in the swarm.
        :param leader_size: Size of the leader archive.
        :param n_evaluations: Total function-evaluation budget.
        :param epsilons: Per-objective ε values for OMOPSO; ``None`` uses SMPSO.
            Length must equal the number of objectives.
        :param seed: Random seed (``None`` for non-deterministic).
        :param record_frequency: History snapshot interval (algorithm iterations).
        :param algorithm_kwargs: Additional keyword arguments forwarded to the
            Platypus algorithm constructor.
        """

        def __init__(
            self,
            swarm_size: int = 100,
            leader_size: int = 100,
            n_evaluations: int = 10_000,
            epsilons: list | None = None,
            seed: int | None = 42,
            record_frequency: int = 1,
            **algorithm_kwargs,
        ) -> None:
            self.swarm_size = swarm_size
            self.leader_size = leader_size
            self.n_evaluations = n_evaluations
            self.epsilons = epsilons
            self.seed = seed
            self.record_frequency = record_frequency
            self.algorithm_kwargs = algorithm_kwargs

        def solve(self, problem: "CalibrationProblem", **kwargs) -> "CalibrationResult":
            """Run the PSO solver and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param problem: Calibration problem wrapping model + data + objectives.
            :param kwargs: Per-call overrides: ``swarm_size``, ``leader_size``,
                ``n_evaluations``, ``seed``, ``record_frequency``, ``epsilons``.
            """
            swarm_size       = kwargs.pop("swarm_size",       self.swarm_size)
            leader_size      = kwargs.pop("leader_size",      self.leader_size)
            n_evaluations    = kwargs.pop("n_evaluations",    self.n_evaluations)
            epsilons         = kwargs.pop("epsilons",         self.epsilons)
            seed             = kwargs.pop("seed",             self.seed)
            record_frequency = kwargs.pop("record_frequency", self.record_frequency)

            if epsilons is not None and len(epsilons) != problem.n_objectives:
                raise ValueError(
                    f"epsilons length ({len(epsilons)}) must match the number of "
                    f"objectives ({problem.n_objectives})."
                )

            algo = platypus.OMOPSO if epsilons is not None else platypus.SMPSO
            extra = {"epsilons": epsilons} if epsilons is not None else {}

            return PlatypusSolver(
                algo,
                swarm_size=swarm_size,
                leader_size=leader_size,
                n_evaluations=n_evaluations,
                seed=seed,
                record_frequency=record_frequency,
                **extra,
                **{**self.algorithm_kwargs, **kwargs},
            ).solve(problem)

except ImportError:

    class _PlatypusSolverStub:
        """Placeholder — requires ``platypus-opt``.

        Install with::

            pip install sparsehydro[platypus]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "platypus-opt is required for PlatypusSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

        def solve(self, problem) -> None:  # type: ignore[return]
            raise ImportError(
                "platypus-opt is required for PlatypusSolver. "
                "Install with: pip install sparsehydro[platypus]"
            )

    PlatypusSolver = _PlatypusSolverStub  # type: ignore[assignment, misc]

    def ParticleSwarmSolver(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "platypus-opt is required for ParticleSwarmSolver. "
            "Install with: pip install sparsehydro[platypus]"
        )
