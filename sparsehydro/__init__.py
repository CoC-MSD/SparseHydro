"""sparsehydro — interfaces and utilities for parsimonious hydrological models."""

from .enums import ModelState
from .parameters import FieldRecord, ScalarParameter, VectorParameter
from .registry import ModelRegistry, registry
from .filters import FilterResult, apply_savgol_filter, compute_thresholds
from .events import EventRecord, detect_events, events_to_dataframe, load_events_from_csv

from .models import IModel, IUnitHydroComponent, EnsembleModel, SeasonalityModel
from .models.rdii import IAModel, RTKTriangle, RTKEnsembleModel, triangular_uh, default_rtk_params, RDIIModel
from .models.unithydrograph import (
    UnitHydrographAdapter,
    create_uh_model,
    register_all_uh_models,
    GammaUH,
    NashUH,
    TriangleUH,
    SequentialFitter,
    SequentialFitSummary,
)

__version__ = "0.1.0"
__all__ = [
    "ModelState",
    "FieldRecord",
    "ScalarParameter",
    "VectorParameter",
    "ModelRegistry",
    "registry",
    "FilterResult",
    "apply_savgol_filter",
    "compute_thresholds",
    "EventRecord",
    "detect_events",
    "events_to_dataframe",
    "load_events_from_csv",
    # models
    "IModel",
    "IUnitHydroComponent",
    "EnsembleModel",
    "SeasonalityModel",
    # rdii
    "IAModel",
    "RTKTriangle",
    "RTKEnsembleModel",
    "triangular_uh",
    "default_rtk_params",
    "RDIIModel",
    # unithydrograph
    "UnitHydrographAdapter",
    "create_uh_model",
    "register_all_uh_models",
    "GammaUH",
    "NashUH",
    "TriangleUH",
    "SequentialFitter",
    "SequentialFitSummary",
    "__version__",
]

try:
    from .models import ITorchModel
    __all__ += ["ITorchModel"]
except ImportError:  # pragma: no cover
    pass

try:
    from .visualization import (
        VisualizationModel,
        plot_timeseries,
        plot_residuals_scatter,
        plot_cumulative_volume,
        plot_data_explorer,
        plot_ensemble_timeseries,
        plot_ensemble_components,
        plot_pareto_evolution,
        plot_parallel_coordinates,
        plot_objective_convergence,
        plot_parameter_distributions,
        plot_sensitivity_heatmap,
        plot_pareto_scatter_matrix,
        plot_rtk_shape,
        plot_rdii_components,
        plot_calibration_dashboard,
        plot_rainfall_flow_with_events,
        plot_filter_signals,
        plot_event_detection,
        plot_sequential_fit,
        plot_parameter_evolution,
        plot_effective_area,
    )
    __all__ += [
        "VisualizationModel",
        "plot_timeseries",
        "plot_residuals_scatter",
        "plot_cumulative_volume",
        "plot_data_explorer",
        "plot_ensemble_timeseries",
        "plot_ensemble_components",
        "plot_pareto_evolution",
        "plot_parallel_coordinates",
        "plot_objective_convergence",
        "plot_parameter_distributions",
        "plot_sensitivity_heatmap",
        "plot_pareto_scatter_matrix",
        "plot_rtk_shape",
        "plot_rdii_components",
        "plot_calibration_dashboard",
        "plot_rainfall_flow_with_events",
        "plot_filter_signals",
        "plot_event_detection",
        "plot_sequential_fit",
        "plot_parameter_evolution",
        "plot_effective_area",
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
