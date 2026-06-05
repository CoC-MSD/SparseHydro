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
        CombinedHydroModel,
        SeasonalityModel,
        compute_time_features,
        peak_weighted_mse,
        nash_sutcliffe,
    )
    __all__ += [
        "IAModel",
        "RTKTriangle",
        "triangular_uh",
        "RDIIModel",
        "CombinedHydroModel",
        "SeasonalityModel",
        "compute_time_features",
        "peak_weighted_mse",
        "nash_sutcliffe",
    ]
except ImportError:  # pragma: no cover
    pass

from .ensemble import EnsembleModel
__all__ += ["EnsembleModel"]

try:
    from .visualization import (
        VisualizationModel,
        plot_timeseries,
        plot_residuals_scatter,
        plot_cumulative_volume,
        plot_data_explorer,
        plot_ensemble_timeseries,
        plot_pareto_evolution,
        plot_parallel_coordinates,
        plot_objective_convergence,
        plot_parameter_distributions,
        plot_sensitivity_heatmap,
        plot_pareto_scatter_matrix,
        plot_rtk_shape,
        plot_rdii_components,
        plot_calibration_dashboard,
    )
    __all__ += [
        "VisualizationModel",
        "plot_timeseries",
        "plot_residuals_scatter",
        "plot_cumulative_volume",
        "plot_data_explorer",
        "plot_ensemble_timeseries",
        "plot_pareto_evolution",
        "plot_parallel_coordinates",
        "plot_objective_convergence",
        "plot_parameter_distributions",
        "plot_sensitivity_heatmap",
        "plot_pareto_scatter_matrix",
        "plot_rtk_shape",
        "plot_rdii_components",
        "plot_calibration_dashboard",
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
