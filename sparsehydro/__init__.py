"""sparsehydro — interfaces and utilities for parsimonious hydrological models."""

from .enums import ModelState
from .interfaces import IModel
from .parameters import ScalarParameter, VectorParameter
from .registry import ModelRegistry, registry
from .unithydrograph import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
)

__version__ = "0.1.0"
__all__ = [
    "ModelState",
    "ScalarParameter",
    "VectorParameter",
    "IModel",
    "ModelRegistry",
    "registry",
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
    "ITorchModel",
    "__version__",
]

try:
    from .torch_model import ITorchModel
except ImportError:  # pragma: no cover
    pass

try:
    from .rdii import (
        IAModel,
        RTKTriangle,
        triangular_uh,
        RDIIModel,
        peak_weighted_mse,
        nash_sutcliffe,
    )
    __all__ += [
        "IAModel",
        "RTKTriangle",
        "triangular_uh",
        "RDIIModel",
        "peak_weighted_mse",
        "nash_sutcliffe",
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from .rdii import RDIIOptimizer
    __all__ += ["RDIIOptimizer"]
except ImportError:  # pragma: no cover
    pass

try:
    from .visualization import (
        VisualizationModel,
        plot_timeseries,
        plot_pareto_evolution,
        plot_parallel_coordinates,
    )
    __all__ += [
        "VisualizationModel",
        "plot_timeseries",
        "plot_pareto_evolution",
        "plot_parallel_coordinates",
    ]
except ImportError:  # pragma: no cover
    pass

try:
    from .calibration import (
        IObjective,
        MSE,
        RMSE,
        MAE,
        PeakWeightedMSE,
        NashSutcliffe,
        KGE,
        CalibrationProblem,
        CalibrationResult,
        GenerationRecord,
        ISolver,
        NSGAIISolver,
        ScipySolver,
    )
    __all__ += [
        "IObjective",
        "MSE",
        "RMSE",
        "MAE",
        "PeakWeightedMSE",
        "NashSutcliffe",
        "KGE",
        "CalibrationProblem",
        "CalibrationResult",
        "GenerationRecord",
        "ISolver",
        "NSGAIISolver",
        "ScipySolver",
    ]
except ImportError:  # pragma: no cover
    pass
