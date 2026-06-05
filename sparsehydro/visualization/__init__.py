"""sparsehydro.visualization — interactive Plotly charts for hydrological models.

Public API
----------
+------------------------------------------+------------------------------------------------------+
| Name                                     | Description                                          |
+==========================================+======================================================+
| :func:`plot_timeseries`                  | Rainfall + observed/predicted dual-axis subplots     |
| :func:`plot_residuals_scatter`           | Obs-vs-pred scatter, residual bars, autocorrelation  |
| :func:`plot_cumulative_volume`           | Cumulative sums + volume error                       |
| :func:`plot_calibration_timeseries`      | 2-row dashboard: exogenous + pred/obs + 1v1 scatter  |
| :class:`VisualizationModel`              | IModel wrapper for time-series plots                 |
+------------------------------------------+------------------------------------------------------+
| :func:`plot_pareto_evolution`            | Animated Pareto front with generation slider         |
| :func:`plot_parallel_coordinates`        | Parallel-axis trade-off explorer                     |
| :func:`plot_objective_convergence`       | Best-value line + percentile band per generation     |
| :func:`plot_parameter_distributions`     | Violin grid of Pareto parameter spread               |
| :func:`plot_sensitivity_heatmap`         | Parameter–objective Pearson correlation heatmap      |
| :func:`plot_pareto_scatter_matrix`       | SPLOM of all objectives                              |
+------------------------------------------+------------------------------------------------------+
| :func:`plot_rtk_shape`                   | Unit hydrograph shape per RTKTriangle                |
| :func:`plot_rdii_components`             | Stacked per-triangle RDII contributions              |
+------------------------------------------+------------------------------------------------------+
| :func:`plot_calibration_dashboard`       | Multi-panel offline HTML dashboard                   |
+------------------------------------------+------------------------------------------------------+

All functions require the optional ``plotly`` dependency::

    pip install plotly

RDII-specific functions additionally require::

    pip install sparsehydro[rdii]
"""

try:
    from .timeseries import (
        VisualizationModel,
        plot_calibration_timeseries,
        plot_cumulative_volume,
        plot_data_explorer,
        plot_ensemble_timeseries,
        plot_residuals_scatter,
        plot_timeseries,
    )
    from .calibration import (
        plot_objective_convergence,
        plot_parallel_coordinates,
        plot_parameter_distributions,
        plot_pareto_evolution,
        plot_pareto_scatter_matrix,
        plot_sensitivity_heatmap,
    )
    from .rdii import plot_rdii_components, plot_rtk_shape
    from .dashboard import plot_calibration_dashboard

except ImportError:

    def plot_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_timeseries. Install with: pip install plotly")

    def plot_residuals_scatter(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_residuals_scatter. Install with: pip install plotly")

    def plot_cumulative_volume(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_cumulative_volume. Install with: pip install plotly")

    def plot_pareto_evolution(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_pareto_evolution. Install with: pip install plotly")

    def plot_parallel_coordinates(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_parallel_coordinates. Install with: pip install plotly")

    def plot_objective_convergence(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_objective_convergence. Install with: pip install plotly")

    def plot_parameter_distributions(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_parameter_distributions. Install with: pip install plotly")

    def plot_sensitivity_heatmap(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_sensitivity_heatmap. Install with: pip install plotly")

    def plot_pareto_scatter_matrix(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_pareto_scatter_matrix. Install with: pip install plotly")

    def plot_rtk_shape(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_rtk_shape. Install with: pip install plotly")

    def plot_rdii_components(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_rdii_components. Install with: pip install plotly")

    def plot_calibration_dashboard(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_calibration_dashboard. Install with: pip install plotly")

    def plot_data_explorer(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_data_explorer. Install with: pip install plotly")

    def plot_ensemble_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_ensemble_timeseries. Install with: pip install plotly")

    from ..interfaces import IModel as _IModel  # noqa: E402

    class VisualizationModel(_IModel):  # type: ignore[no-redef]
        model_name = "visualization"

        def initialize(self) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def validate(self) -> bool:  # type: ignore[return]
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def prepare(self, data, **kwargs) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def predict(self):  # type: ignore[return]
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")

        def finalize(self) -> None:
            raise ImportError("plotly is required for VisualizationModel. Install with: pip install plotly")


__all__ = [
    # Time-domain
    "plot_timeseries",
    "plot_residuals_scatter",
    "plot_cumulative_volume",
    "plot_data_explorer",
    "plot_ensemble_timeseries",
    "VisualizationModel",
    # Calibration
    "plot_pareto_evolution",
    "plot_parallel_coordinates",
    "plot_objective_convergence",
    "plot_parameter_distributions",
    "plot_sensitivity_heatmap",
    "plot_pareto_scatter_matrix",
    # RDII
    "plot_rtk_shape",
    "plot_rdii_components",
    # Dashboard
    "plot_calibration_dashboard",
]
