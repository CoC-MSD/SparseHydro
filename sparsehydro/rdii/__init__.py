"""RDII subpackage for sparsehydro.

Physics-based Rainfall-Derived Inflow and Infiltration model combining:

- Exponential Initial Abstraction recovery/depletion driven by temperature.
- An arbitrary number N of triangular RTK unit hydrographs.
- NSGA-II multi-objective calibration (peak-weighted MSE vs NSE).
- Interactive Plotly visualizations (time series, Pareto evolution,
  parallel coordinates).

Basic usage::

    from sparsehydro.rdii import RDIIModel

    model = RDIIModel(n_triangles=3)
    model.initialize()
    model.validate()
    model.prepare(df)        # DataFrame: datetime, rainfall_mm [, flow_cfs, temperature_c]
    result = model.predict() # DataFrame: datetime, rdii_mm, p_excess_mm
    model.finalize()

Multi-objective calibration (requires ``pip install sparsehydro[rdii]``)::

    from sparsehydro.rdii import RDIIOptimizer, plot_pareto_evolution

    opt = RDIIOptimizer(model, observed_flow=df["flow_cfs"].to_numpy())
    rdii_result = opt.run(pop_size=100, n_gen=200)
    fig = plot_pareto_evolution(rdii_result)
    fig.show()
"""

from .initial_abstraction import IAModel
from .model import RDIIModel
from .objectives import nash_sutcliffe, peak_weighted_mse
from .rtk_triangle import RTKTriangle, triangular_uh

__all__ = [
    "IAModel",
    "RTKTriangle",
    "triangular_uh",
    "RDIIModel",
    "peak_weighted_mse",
    "nash_sutcliffe",
]

try:
    from .optimization import GenerationRecord, RDIIOptimizer, RDIIResult

    __all__ += ["RDIIOptimizer", "RDIIResult", "GenerationRecord"]
except ImportError:  # pragma: no cover
    pass

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
