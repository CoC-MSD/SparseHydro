"""Time-domain interactive plots and VisualizationModel.

Public API (all conditional on plotly being installed):

- :func:`plot_timeseries`        — rainfall + observed/predicted subplots
- :func:`plot_residuals_scatter` — observed-vs-predicted, residual bars, autocorrelation
- :func:`plot_cumulative_volume` — cumulative sums + volume error
- :class:`VisualizationModel`    — IModel wrapper for the above plots
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ..enums import ModelState
from ..interfaces import IModel

if TYPE_CHECKING:
    from ..calibration.result import CalibrationResult

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    # ------------------------------------------------------------------
    # plot_timeseries
    # ------------------------------------------------------------------

    def plot_timeseries(
        datetime,
        rainfall_mm: np.ndarray,
        observed_flow,
        predicted_flow: np.ndarray,
        title: str = "Model Time Series",
        rainfall_label: str = "Rainfall [mm]",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Dual-axis time series: rainfall (top) and flow (bottom).

        The rainfall subplot uses an **inverted Y-axis** (bars grow downward)
        and shares the X-axis with the flow subplot.

        :param datetime: Time axis values (array-like of dates or timestamps).
        :param rainfall_mm: Rainfall depth per time step [mm].
        :type rainfall_mm: numpy.ndarray
        :param observed_flow: Observed flow values, or ``None`` to omit.
        :param predicted_flow: Model-predicted output [same units as observed].
        :type predicted_flow: numpy.ndarray
        :param title: Figure title.
        :param rainfall_label: Y-axis label for the rainfall panel.
        :param flow_label: Y-axis label for the flow panel.
        :returns: Plotly Figure with 2 rows, shared X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.3, 0.7],
            vertical_spacing=0.04,
            subplot_titles=["Rainfall", "Flow"],
        )

        fig.add_trace(
            go.Bar(
                x=datetime,
                y=rainfall_mm,
                name=rainfall_label,
                marker_color="steelblue",
                opacity=0.7,
            ),
            row=1,
            col=1,
        )
        fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)

        if observed_flow is not None:
            fig.add_trace(
                go.Scatter(
                    x=datetime,
                    y=np.asarray(observed_flow),
                    name="Observed",
                    mode="lines",
                    line=dict(color="black", width=1.5),
                ),
                row=2,
                col=1,
            )

        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=np.asarray(predicted_flow),
                name="Predicted",
                mode="lines",
                line=dict(color="crimson", width=2, dash="dash"),
            ),
            row=2,
            col=1,
        )
        fig.update_yaxes(title_text=flow_label, row=2, col=1)
        fig.update_xaxes(title_text="Date / Time", row=2, col=1)
        fig.update_layout(
            title=title,
            height=600,
            legend=dict(x=0.01, y=0.45, bgcolor="rgba(255,255,255,0.7)"),
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_residuals_scatter
    # ------------------------------------------------------------------

    def plot_residuals_scatter(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Residual Diagnostics",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Three-panel residual diagnostic plot.

        **Row 1** — Observed vs predicted scatter with a 1:1 diagonal line.

        **Row 2** — Residual (observed − predicted) bar chart, coloured red
        (over-prediction) / blue (under-prediction).

        **Row 3** — Residual autocorrelation bars at lags 0–30, showing
        systematic bias structure.

        :param datetime: Time axis for the residual bar chart.
        :param observed: Observed values.
        :type observed: numpy.ndarray
        :param predicted: Predicted values.
        :type predicted: numpy.ndarray
        :param title: Figure title.
        :param flow_label: Axis label used for observed / predicted axes.
        :returns: Plotly Figure with 3 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        residuals = obs - pred

        max_lags = min(30, len(residuals) - 1)
        std = np.std(residuals)
        if std == 0:
            acf = np.zeros(max_lags + 1)
        else:
            full = np.correlate(residuals - residuals.mean(), residuals - residuals.mean(), mode="full")
            full /= full[len(residuals) - 1]
            acf = full[len(residuals) - 1 : len(residuals) + max_lags]

        colors = ["crimson" if r < 0 else "steelblue" for r in residuals]

        fig = make_subplots(
            rows=3,
            cols=1,
            row_heights=[0.35, 0.35, 0.3],
            vertical_spacing=0.08,
            subplot_titles=[
                "Observed vs Predicted",
                f"Residuals ({flow_label})",
                "Residual Autocorrelation",
            ],
        )

        # Row 1 — observed vs predicted
        lo = min(obs.min(), pred.min())
        hi = max(obs.max(), pred.max())
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color="grey", dash="dot", width=1),
                name="1:1 line",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=obs,
                y=pred,
                mode="markers",
                marker=dict(color="steelblue", size=5, opacity=0.6),
                name="Solutions",
                showlegend=False,
            ),
            row=1, col=1,
        )
        fig.update_xaxes(title_text=f"Observed {flow_label}", row=1, col=1)
        fig.update_yaxes(title_text=f"Predicted {flow_label}", row=1, col=1)

        # Row 2 — residual bars
        fig.add_trace(
            go.Bar(
                x=datetime,
                y=residuals,
                marker_color=colors,
                name="Residual",
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1), row=2, col=1)
        fig.update_yaxes(title_text="Obs − Pred", row=2, col=1)

        # Row 3 — autocorrelation
        lags = list(range(max_lags + 1))
        acf_colors = ["crimson" if a < 0 else "steelblue" for a in acf]
        fig.add_trace(
            go.Bar(x=lags, y=acf, marker_color=acf_colors, name="ACF", showlegend=False),
            row=3, col=1,
        )
        # 95% confidence bounds (±1.96/sqrt(n))
        ci = 1.96 / np.sqrt(len(residuals))
        fig.add_hline(y=ci, line=dict(color="grey", dash="dash", width=1), row=3, col=1)
        fig.add_hline(y=-ci, line=dict(color="grey", dash="dash", width=1), row=3, col=1)
        fig.update_xaxes(title_text="Lag", row=3, col=1)
        fig.update_yaxes(title_text="Autocorrelation", row=3, col=1)

        fig.update_layout(
            title=title,
            height=800,
            hovermode="x unified",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_cumulative_volume
    # ------------------------------------------------------------------

    def plot_cumulative_volume(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        title: str = "Cumulative Volume",
        flow_label: str = "Flow",
    ) -> go.Figure:
        """Two-panel cumulative volume comparison.

        **Row 1** — Cumulative sums of observed and predicted with a shaded
        fill between the two lines, highlighting volume discrepancy.

        **Row 2** — Cumulative volume error (obs_cumsum − pred_cumsum) with a
        zero-reference line.

        :param datetime: Time axis values.
        :param observed: Observed values.
        :type observed: numpy.ndarray
        :param predicted: Predicted values.
        :type predicted: numpy.ndarray
        :param title: Figure title.
        :param flow_label: Units label for volume axis.
        :returns: Plotly Figure with 2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        obs = np.asarray(observed, dtype=float)
        pred = np.asarray(predicted, dtype=float)
        cum_obs = np.cumsum(obs)
        cum_pred = np.cumsum(pred)
        cum_error = cum_obs - cum_pred

        dt = list(datetime)

        fig = make_subplots(
            rows=2,
            cols=1,
            shared_xaxes=True,
            row_heights=[0.6, 0.4],
            vertical_spacing=0.06,
            subplot_titles=["Cumulative Volume", "Cumulative Volume Error"],
        )

        # Row 1 — cumulative sums + fill between
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_obs,
                mode="lines",
                name="Observed cumulative",
                line=dict(color="black", width=2),
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_pred,
                mode="lines",
                name="Predicted cumulative",
                fill="tonexty",
                fillcolor="rgba(220,50,50,0.15)",
                line=dict(color="crimson", width=2, dash="dash"),
            ),
            row=1, col=1,
        )
        fig.update_yaxes(title_text=f"Cumulative {flow_label}", row=1, col=1)

        # Row 2 — volume error
        err_color = np.where(cum_error >= 0, "steelblue", "crimson").tolist()
        fig.add_trace(
            go.Scatter(
                x=dt,
                y=cum_error,
                mode="lines",
                name="Volume error",
                line=dict(color="steelblue", width=2),
                showlegend=False,
            ),
            row=2, col=1,
        )
        fig.add_hline(y=0, line=dict(color="black", width=1, dash="dot"), row=2, col=1)
        fig.update_yaxes(title_text="Obs − Pred (cumul.)", row=2, col=1)
        fig.update_xaxes(title_text="Date / Time", row=2, col=1)

        fig.update_layout(
            title=title,
            height=600,
            hovermode="x unified",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.7)"),
        )
        return fig

    # ======================================================================
    # VisualizationModel
    # ======================================================================

    class VisualizationModel(IModel):
        """General-purpose visualization model for any :class:`~sparsehydro.interfaces.IModel` output.

        Wraps :func:`plot_timeseries`, :func:`plot_pareto_evolution`, and
        :func:`plot_parallel_coordinates` behind the standard
        :class:`~sparsehydro.interfaces.IModel` lifecycle.

        Column names for the input DataFrame are specified at :meth:`prepare`
        time, making this model compatible with any model's output format.

        :meth:`predict` returns the input DataFrame unchanged so the model can
        be composed in a pipeline.  Figures are available as read-only
        properties afterward::

            viz.timeseries_figure    # always generated
            viz.pareto_figure        # generated when calibration_result is supplied
            viz.parallel_figure      # generated when calibration_result is supplied

        :param title: Base title used across all figures.
        :param rainfall_label: Y-axis label for the rainfall panel.
        :param flow_label: Y-axis label for the flow panel.
        """

        model_name = "visualization"

        def __init__(
            self,
            title: str = "Model Time Series",
            rainfall_label: str = "Rainfall [mm]",
            flow_label: str = "Flow",
        ) -> None:
            super().__init__()
            self._title = title
            self._rainfall_label = rainfall_label
            self._flow_label = flow_label
            self._data: pd.DataFrame | None = None
            self._datetime_col: str = "datetime"
            self._predicted_col: str = "predicted"
            self._observed_col: str | None = None
            self._rainfall_col: str | None = None
            self._calibration_result: "CalibrationResult | None" = None
            self._timeseries_fig: go.Figure | None = None
            self._pareto_fig: go.Figure | None = None
            self._parallel_fig: go.Figure | None = None

        def initialize(self) -> None:
            """Advance to INITIALIZED."""
            self._state = ModelState.INITIALIZED

        def validate(self) -> bool:
            """Validate and advance to VALIDATED.

            :returns: Always ``True``.
            """
            self._state = ModelState.VALIDATED
            return True

        def prepare(
            self,
            data: pd.DataFrame,
            datetime_col: str = "datetime",
            predicted_col: str = "predicted",
            observed_col: str | None = None,
            rainfall_col: str | None = None,
            calibration_result: "CalibrationResult | None" = None,
        ) -> None:
            """Load model output data and configure column mapping.

            :param data: Any model's output DataFrame.
            :param datetime_col: Column name for the time axis.
            :param predicted_col: Column name for the predicted output signal.
            :param observed_col: Column name for observed data, or ``None`` to omit.
            :param rainfall_col: Column name for rainfall forcing, or ``None`` to omit.
            :param calibration_result: When provided, :meth:`predict` also
                generates Pareto-evolution and parallel-coordinates figures.
            :raises ValueError: If required columns are absent.
            """
            missing = [c for c in (datetime_col, predicted_col) if c not in data.columns]
            if missing:
                raise ValueError(f"prepare() data is missing required columns: {missing}")
            self._data = data.copy()
            self._datetime_col = datetime_col
            self._predicted_col = predicted_col
            self._observed_col = observed_col
            self._rainfall_col = rainfall_col
            self._calibration_result = calibration_result
            self._timeseries_fig = None
            self._pareto_fig = None
            self._parallel_fig = None
            self._state = ModelState.PREPARED

        def predict(self) -> pd.DataFrame:
            """Generate all applicable figures and return the input DataFrame.

            :returns: The DataFrame supplied to :meth:`prepare` (unchanged).
            :raises RuntimeError: If :meth:`prepare` has not been called.
            """
            if self._data is None:
                raise RuntimeError("Call prepare(data) before predict().")

            from . import plot_pareto_evolution, plot_parallel_coordinates

            df = self._data
            datetime = df[self._datetime_col]
            predicted = df[self._predicted_col].to_numpy(dtype=float)
            observed = (
                df[self._observed_col].to_numpy(dtype=float)
                if self._observed_col and self._observed_col in df.columns
                else None
            )
            rainfall = (
                df[self._rainfall_col].to_numpy(dtype=float)
                if self._rainfall_col and self._rainfall_col in df.columns
                else np.zeros(len(df))
            )

            self._timeseries_fig = plot_timeseries(
                datetime=datetime,
                rainfall_mm=rainfall,
                observed_flow=observed,
                predicted_flow=predicted,
                title=self._title,
                rainfall_label=self._rainfall_label,
                flow_label=self._flow_label,
            )

            if self._calibration_result is not None:
                self._pareto_fig = plot_pareto_evolution(
                    self._calibration_result,
                    title=f"{self._title} — Pareto Front Evolution",
                )
                self._parallel_fig = plot_parallel_coordinates(
                    self._calibration_result,
                    title=f"{self._title} — Parallel Coordinates",
                )

            self._state = ModelState.PREDICTED
            return df

        def finalize(self) -> None:
            """Release stored data and figures and advance to FINALIZED."""
            self._data = None
            self._calibration_result = None
            self._timeseries_fig = None
            self._pareto_fig = None
            self._parallel_fig = None
            self._state = ModelState.FINALIZED

        @property
        def timeseries_figure(self) -> "go.Figure | None":
            """Dual-axis rainfall/flow time series, or ``None`` before :meth:`predict`."""
            return self._timeseries_fig

        @property
        def pareto_figure(self) -> "go.Figure | None":
            """Animated Pareto-front evolution, or ``None`` when no result was supplied."""
            return self._pareto_fig

        @property
        def parallel_figure(self) -> "go.Figure | None":
            """Parallel-coordinates trade-off explorer, or ``None`` when no result was supplied."""
            return self._parallel_fig

except ImportError:

    def plot_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_timeseries. Install with: pip install plotly")

    def plot_residuals_scatter(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_residuals_scatter. Install with: pip install plotly")

    def plot_cumulative_volume(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_cumulative_volume. Install with: pip install plotly")

    class VisualizationModel(IModel):  # type: ignore[no-redef]
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
