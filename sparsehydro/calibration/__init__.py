"""sparsehydro.calibration — general-purpose model calibration abstractions.

Public API
----------

**Objectives**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`IObjective`
     - Abstract objective function interface
   * - :class:`MSE`
     - Mean squared error
   * - :class:`RMSE`
     - Root mean squared error
   * - :class:`MAE`
     - Mean absolute error
   * - :class:`PeakWeightedMSE`
     - Flow-weighted MSE
   * - :class:`NashSutcliffe`
     - Nash-Sutcliffe efficiency
   * - :class:`KGE`
     - Kling-Gupta efficiency
   * - :class:`PBIAS`
     - Absolute percent bias
   * - :class:`VolumeRelativeError`
     - Relative total volume error
   * - :class:`LogNSE`
     - NSE on log-transformed flows
   * - :class:`IndexOfAgreement`
     - Willmott's index of agreement
   * - :class:`ConcordanceCorrelationCoefficient`
     - Lin's concordance correlation coefficient

**Problem & results**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`CalibrationProblem`
     - Wraps any IModel + observed + objectives
   * - :class:`CalibrationResult`
     - Pareto front + generation history
   * - :class:`GenerationRecord`
     - Per-generation population snapshot

**Solvers**

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Name
     - Description
   * - :class:`ISolver`
     - Abstract solver interface
   * - :class:`NSGAIISolver`
     - NSGA-II (requires pymoo)
   * - :class:`ParticleSwarmSolver`
     - SMPSO / OMOPSO (requires platypus-opt)
   * - :class:`PlatypusSolver`
     - Any Platypus algorithm (requires platypus-opt)
   * - :class:`ScipySolver`
     - SciPy single-objective (requires scipy)
"""

from .callbacks import ProgressCallback
from .objectives import (
    IObjective,
    KGE,
    MAE,
    MSE,
    RMSE,
    NashSutcliffe,
    PeakWeightedMSE,
    PBIAS,
    VolumeRelativeError,
    LogNSE,
    IndexOfAgreement,
    ConcordanceCorrelationCoefficient,
)
from .problem import CalibrationProblem
from .result import CalibrationResult, GenerationRecord, _identify_pareto
from .solvers import ISolver, NSGAIISolver, ParticleSwarmSolver, PlatypusSolver, ScipySolver

__all__ = [
    # Objectives
    "IObjective",
    "MSE",
    "RMSE",
    "MAE",
    "PeakWeightedMSE",
    "NashSutcliffe",
    "KGE",
    "PBIAS",
    "VolumeRelativeError",
    "LogNSE",
    "IndexOfAgreement",
    "ConcordanceCorrelationCoefficient",
    # Problem
    "CalibrationProblem",
    # Result
    "CalibrationResult",
    "GenerationRecord",
    # Solvers
    "ISolver",
    "NSGAIISolver",
    "ParticleSwarmSolver",
    "PlatypusSolver",
    "ScipySolver",
    # Callbacks
    "ProgressCallback",
]
