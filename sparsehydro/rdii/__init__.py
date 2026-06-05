"""RDII subpackage for sparsehydro.

Physics-based Rainfall-Derived Inflow and Infiltration model combining:

- Exponential Initial Abstraction recovery/depletion driven by temperature.
- An arbitrary number N of triangular RTK unit hydrographs.
- Interactive Plotly visualizations (time series, Pareto evolution,
  parallel coordinates).

Basic usage::

    from sparsehydro.rdii import RDIIModel

    model = RDIIModel(n_triangles=3)
    model.initialize()
    model.validate()
    model.prepare(df)        # DataFrame: datetime, rainfall_mm [, flow_cfs, temperature_c]
    result = model.predict() # DataFrame: datetime, rdii_cfs, rdii_mm, p_excess_mm
    model.finalize()

Calibration (requires ``pip install sparsehydro[rdii]``)::

    from sparsehydro.rdii import RDIIModel
    from sparsehydro.calibration import (
        CalibrationProblem, NSGAIISolver, PeakWeightedMSE, NashSutcliffe,
    )

    model = RDIIModel(n_triangles=3)
    model.initialize()
    model.validate()

    problem = CalibrationProblem(
        model=model,
        data=df,
        objectives=[PeakWeightedMSE(), NashSutcliffe()],
        column_map={
            "rainfall_mm":  "rain",
            "observed":     "flow_cfs",
            "predicted":    "rdii_cfs",
        },
    )
    result = NSGAIISolver(pop_size=100, n_gen=200).solve(problem)
"""

from .combined_model import CombinedHydroModel
from .initial_abstraction import IAModel
from .model import RDIIModel
from .objectives import nash_sutcliffe, peak_weighted_mse
from .rtk_triangle import RTKTriangle, triangular_uh
from .seasonality import SeasonalityModel, compute_time_features

__all__ = [
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

try:
    from .visualization import (
        plot_parallel_coordinates,
        plot_pareto_evolution,
        plot_timeseries,
    )
    from ..visualization import VisualizationModel

    __all__ += [
        "plot_timeseries",
        "plot_pareto_evolution",
        "plot_parallel_coordinates",
        "VisualizationModel",
    ]
except ImportError:  # pragma: no cover
    pass
