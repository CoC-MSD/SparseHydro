"""Multi-objective NSGA-II optimizer for RDIIModel calibration.

:class:`RDIIOptimizer` is a thin adapter over
:class:`~sparsehydro.calibration.problem.CalibrationProblem` and
:class:`~sparsehydro.calibration.solvers.NSGAIISolver`.  It pre-configures
two objectives — peak-weighted MSE (minimised) and Nash–Sutcliffe efficiency
(maximised) — and returns a general
:class:`~sparsehydro.calibration.result.CalibrationResult`.

Requires the optional ``pymoo`` dependency::

    pip install sparsehydro[rdii]
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .model import RDIIModel

# Re-export GenerationRecord and _identify_pareto for backward compatibility
# and for use by visualization.py
from ..calibration.result import (  # noqa: F401
    CalibrationResult,
    GenerationRecord,
    _identify_pareto,
)

try:
    from ..calibration import (
        CalibrationProblem,
        NashSutcliffe,
        PeakWeightedMSE,
    )
    from ..calibration.solvers.nsga2 import NSGAIISolver

    class RDIIOptimizer:
        """Multi-objective NSGA-II calibrator for :class:`~sparsehydro.rdii.model.RDIIModel`.

        Automatically discovers parameter names and bounds from the model's
        :class:`~sparsehydro.parameters.ScalarParameter` registry and
        delegates to :class:`~sparsehydro.calibration.solvers.NSGAIISolver`
        using :class:`~sparsehydro.calibration.objectives.PeakWeightedMSE` and
        :class:`~sparsehydro.calibration.objectives.NashSutcliffe` as objectives.

        :param model: A prepared ``RDIIModel`` (``model.is_prepared() == True``).
        :type model: RDIIModel
        :param observed_flow: 1-D array of observed RDII/flow values aligned
            with the model's prepared time series.
        :type observed_flow: numpy.ndarray
        :raises RuntimeError: If ``model`` is not in PREPARED state.
        """

        def __init__(
            self,
            model: "RDIIModel",
            observed_flow: np.ndarray,
        ) -> None:
            if not model.is_prepared():
                raise RuntimeError(
                    "model must be in PREPARED state before creating "
                    "RDIIOptimizer. Call model.prepare(data) first."
                )
            self._model = model
            self._observed = np.asarray(observed_flow, dtype=float)

        def run(
            self,
            pop_size: int = 100,
            n_gen: int = 200,
            seed: int | None = 42,
            verbose: bool = False,
            n_jobs: int = 1,
        ) -> CalibrationResult:
            """Run NSGA-II and return a :class:`~sparsehydro.calibration.result.CalibrationResult`.

            :param pop_size: Population size per generation.
            :param n_gen: Number of generations.
            :param seed: Random seed for reproducibility.
            :param verbose: Print pymoo iteration output.
            :param n_jobs: Number of parallel workers.
            :returns: Calibration result with Pareto front and full history.
            :rtype: CalibrationResult
            """
            problem = CalibrationProblem(
                model=self._model,
                observed=self._observed,
                objectives=[PeakWeightedMSE(), NashSutcliffe()],
                result_extractor=lambda df: df["rdii_mm"].to_numpy(dtype=float),
            )
            solver = NSGAIISolver(
                pop_size=pop_size,
                n_gen=n_gen,
                seed=seed,
                verbose=verbose,
                n_jobs=n_jobs,
            )
            return solver.solve(problem)

except ImportError:

    class RDIIOptimizer:  # type: ignore[no-redef]
        """Placeholder — requires ``pymoo``.

        Install with::

            pip install sparsehydro[rdii]
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ImportError(
                "pymoo is required for RDIIOptimizer. "
                "Install with: pip install sparsehydro[rdii]"
            )
