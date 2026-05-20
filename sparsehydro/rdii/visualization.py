"""Re-exports from :mod:`sparsehydro.visualization` for backward compatibility."""

from ..visualization import (
    plot_parallel_coordinates,
    plot_pareto_evolution,
    plot_timeseries,
)

__all__ = [
    "plot_timeseries",
    "plot_pareto_evolution",
    "plot_parallel_coordinates",
]
