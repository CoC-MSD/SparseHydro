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

    # ------------------------------------------------------------------
    # plot_calibration_timeseries
    # ------------------------------------------------------------------

    def plot_calibration_timeseries(
        datetime,
        observed: np.ndarray,
        predicted: np.ndarray,
        exogenous: "dict[str, tuple[np.ndarray, str]] | None" = None,
        pareto_predictions: "np.ndarray | None" = None,
        confidence_percentiles: tuple = (25, 75),
        tolerance_angles: "list[float] | None" = None,
        rainfall_label: str = "Rainfall",
        observed_label: str = "Observed",
        predicted_label: str = "Predicted",
        title: str = "Calibration Time Series",
    ) -> "go.Figure":
        """Two-row calibration dashboard with linked time axes.

        **Row 1 (full width)** — Exogenous inputs.  All entries in *exogenous*
        that share the same ``units`` string are overlaid on one y-axis; each
        unique unit gets its own secondary/tertiary axis.  Traces whose label
        contains "rain" or "precip" (case-insensitive) are rendered as **inverted
        bars** (downward); everything else is a line.

        **Row 2 Left** — Predicted vs Observed time series.  If
        *pareto_predictions* is supplied, a shaded IQR band (between
        *confidence_percentiles*) is drawn around the predicted line.

        **Row 2 Right** — 1v1 scatter of observed vs predicted.  When
        *pareto_predictions* is supplied, vertical box-whiskers summarise the
        Pareto range at each time step.  A dashed 45° "perfect fit" line is
        always drawn.  Optional *tolerance_angles* draw additional lines
        radiating from the origin at ``45° ± θ``.

        The x-axes of row 1 and row 2-left are **linked**: zooming or panning
        either panel pans the other.

        :param datetime: 1-D datetime-like array aligned with *observed* /
            *predicted*.
        :param observed: 1-D array of observed values.
        :param predicted: 1-D array of predicted values (best solution).
        :param exogenous: ``{label: (values_array, units_string)}`` mapping.
            Traces with the same *units_string* share a y-axis.  Rainfall-like
            traces (label contains "rain" / "precip") are plotted as inverted bars.
        :param pareto_predictions: Array of shape ``(n_solutions, n_timesteps)``
            containing all Pareto-front predictions.  Used for the IQR band and
            scatter box-whiskers.
        :param confidence_percentiles: ``(lower_pct, upper_pct)`` for the IQR band.
        :param tolerance_angles: Angles in degrees from the 45° line.  For each
            ``θ`` two lines are drawn at slopes ``tan(45° ± θ)``.
        :param rainfall_label: String to match when auto-detecting rainfall traces.
        :param title: Figure title.
        :returns: Plotly Figure.
        """
        import math

        observed  = np.asarray(observed,  dtype=float)
        predicted = np.asarray(predicted, dtype=float)
        dt_series = pd.to_datetime(datetime)

        exogenous = exogenous or {}

        # ── Build subplot grid ───────────────────────────────────────────
        fig = make_subplots(
            rows=2, cols=2,
            column_widths=[0.65, 0.35],
            row_heights=[0.35, 0.65],
            specs=[
                [{"colspan": 2}, None],
                [{},             {}],
            ],
            shared_xaxes=False,
            vertical_spacing=0.08,
            horizontal_spacing=0.08,
            subplot_titles=["Exogenous Inputs", "Predicted vs Observed", "1:1 Scatter"],
        )

        # ── Row 1: exogenous traces ──────────────────────────────────────
        # Group by units → shared axis index
        unit_to_axis: dict[str, int] = {}
        axis_idx = 1  # yaxis on row-1 subplot; extra axes are overlaid

        for label, (values, units) in exogenous.items():
            values = np.asarray(values, dtype=float)
            is_rain = any(kw in label.lower() for kw in ("rain", "precip")) or \
                      any(kw in rainfall_label.lower() for kw in ("rain", "precip"))

            if units not in unit_to_axis:
                unit_to_axis[units] = axis_idx
                axis_idx += 1

            yref = "y" if unit_to_axis[units] == 1 else f"y{unit_to_axis[units]}"

            if is_rain:
                fig.add_trace(
                    go.Bar(
                        x=dt_series, y=-values,
                        name=label,
                        yaxis=yref,
                        marker_opacity=0.7,
                        showlegend=True,
                    ),
                    row=1, col=1,
                )
            else:
                fig.add_trace(
                    go.Scatter(
                        x=dt_series, y=values,
                        mode="lines", name=label,
                        yaxis=yref,
                    ),
                    row=1, col=1,
                )

        # ── Row 2 Left: predicted vs observed time series ────────────────
        if pareto_predictions is not None:
            pp = np.asarray(pareto_predictions, dtype=float)
            lo = np.percentile(pp, confidence_percentiles[0], axis=0)
            hi = np.percentile(pp, confidence_percentiles[1], axis=0)
            fig.add_trace(
                go.Scatter(
                    x=pd.concat([pd.Series(dt_series), pd.Series(dt_series[::-1])]),
                    y=np.concatenate([hi, lo[::-1]]),
                    fill="toself",
                    fillcolor="rgba(31,119,180,0.15)",
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=True,
                    name=f"IQR ({confidence_percentiles[0]}–{confidence_percentiles[1]}%)",
                ),
                row=2, col=1,
            )

        fig.add_trace(
            go.Scatter(x=dt_series, y=observed,  mode="lines",
                       name=observed_label,  line=dict(color="#2ca02c")),
            row=2, col=1,
        )
        fig.add_trace(
            go.Scatter(x=dt_series, y=predicted, mode="lines",
                       name=predicted_label, line=dict(color="#1f77b4", dash="dash")),
            row=2, col=1,
        )

        # ── Row 2 Right: 1v1 scatter ─────────────────────────────────────
        max_val = float(np.nanmax([observed, predicted]))

        # Pareto box-whiskers
        if pareto_predictions is not None:
            pp = np.asarray(pareto_predictions, dtype=float)
            for t in range(len(observed)):
                col_vals = pp[:, t]
                fig.add_trace(
                    go.Box(
                        x=[observed[t]] * len(col_vals),
                        y=col_vals,
                        marker_color="rgba(31,119,180,0.3)",
                        line_color="rgba(31,119,180,0.5)",
                        showlegend=False,
                        boxpoints=False,
                    ),
                    row=2, col=2,
                )

        # Best solution markers
        fig.add_trace(
            go.Scatter(
                x=observed, y=predicted,
                mode="markers",
                name="Best solution",
                marker=dict(color="#1f77b4", size=5, opacity=0.7),
            ),
            row=2, col=2,
        )

        # 45° perfect-fit line
        fig.add_trace(
            go.Scatter(
                x=[0, max_val], y=[0, max_val],
                mode="lines",
                name="Perfect fit (45°)",
                line=dict(color="black", dash="dash", width=1.5),
            ),
            row=2, col=2,
        )

        # Tolerance angle lines
        tolerance_angles = tolerance_angles or []
        colors = ["#d62728", "#ff7f0e", "#9467bd", "#8c564b", "#e377c2"]
        for i, theta in enumerate(tolerance_angles):
            color = colors[i % len(colors)]
            for sign, label_suffix in [(+1, "+"), (-1, "−")]:
                slope = math.tan(math.radians(45 + sign * theta))
                fig.add_trace(
                    go.Scatter(
                        x=[0, max_val],
                        y=[0, slope * max_val],
                        mode="lines",
                        name=f"45°{label_suffix}{theta}°",
                        line=dict(color=color, dash="dot", width=1),
                    ),
                    row=2, col=2,
                )

        # ── Link x-axes of row-1 and row-2-left ──────────────────────────
        fig.update_layout(
            title_text=title,
            xaxis2=dict(matches="x"),
            barmode="overlay",
            legend=dict(groupclick="toggleitem"),
            height=700,
        )
        fig.update_yaxes(title_text="", row=2, col=2)
        fig.update_xaxes(title_text="Observed", row=2, col=2)
        fig.update_yaxes(title_text="Predicted", row=2, col=2)

        return fig

    # ------------------------------------------------------------------
    # plot_data_explorer
    # ------------------------------------------------------------------

    def plot_data_explorer(
        df: pd.DataFrame,
        event_id_col: str = "event_id",
        datetime_col: str = "datetime",
        train_event_ids=None,
        val_event_ids=None,
        variables=None,
        rainfall_cols=None,
        title: str = "Data Explorer",
    ) -> "go.Figure":
        """Interactive multi-panel data explorer with train/validation event highlighting.

        One subplot row is created per variable in *variables*, all sharing the
        same X-axis.  Rainfall columns are rendered as **inverted bar charts**
        (bars grow downward); all other variables are rendered as lines.  When
        *train_event_ids* or *val_event_ids* are supplied, the corresponding
        event time spans are shaded in blue and orange respectively.

        :param df: Input DataFrame containing a datetime column and at least one
            numeric variable column.
        :param event_id_col: Column name that identifies individual events.
        :param datetime_col: Column name for the datetime axis.
        :param train_event_ids: Iterable of event IDs to shade as training data
            (blue, ``rgba(0,120,215,0.10)``).
        :param val_event_ids: Iterable of event IDs to shade as validation data
            (orange, ``rgba(230,115,0,0.10)``).
        :param variables: ``{col_name: (axis_label, units)}`` mapping describing
            which columns to plot and how to label them.  When ``None`` all
            numeric columns (excluding *event_id_col* and *datetime_col*) are
            included with their column names as labels.
        :param rainfall_cols: Column names that should be rendered as inverted
            bar charts (rainfall-style).
        :param title: Figure title.
        :returns: Plotly Figure with *N* rows (one per variable), shared X-axis.
        :rtype: plotly.graph_objects.Figure
        """
        if variables is None:
            skip = {event_id_col, datetime_col}
            variables = {
                col: (col, "")
                for col in df.columns
                if col not in skip and pd.api.types.is_numeric_dtype(df[col])
            }

        rainfall_set = set(rainfall_cols or [])
        col_names = list(variables.keys())
        n = len(col_names)
        if n == 0:
            raise ValueError("plot_data_explorer: no numeric variables found to plot.")

        # Row heights: rainfall rows 20%, other rows share remaining equally
        rain_rows = [c in rainfall_set for c in col_names]
        n_rain = sum(rain_rows)
        n_other = n - n_rain
        rain_h = 0.20
        other_h = (1.0 - n_rain * rain_h) / max(n_other, 1)
        heights = [rain_h if r else other_h for r in rain_rows]

        subplot_titles = [
            f"{variables[c][0]} [{variables[c][1]}]" if variables[c][1] else variables[c][0]
            for c in col_names
        ]

        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            row_heights=heights,
            vertical_spacing=0.04,
            subplot_titles=subplot_titles,
        )

        dt = df[datetime_col]

        # ── Variable traces ──────────────────────────────────────────────
        for i, col in enumerate(col_names, start=1):
            label, units = variables[col]
            y_title = f"{label} [{units}]" if units else label

            if col in rainfall_set:
                fig.add_trace(
                    go.Bar(
                        x=dt,
                        y=df[col],
                        name=label,
                        marker_color="steelblue",
                        opacity=0.7,
                    ),
                    row=i, col=1,
                )
                fig.update_yaxes(autorange="reversed", title_text=y_title, row=i, col=1)
            else:
                fig.add_trace(
                    go.Scatter(x=dt, y=df[col], mode="lines", name=label),
                    row=i, col=1,
                )
                fig.update_yaxes(title_text=y_title, row=i, col=1)

        fig.update_xaxes(title_text="Date / Time", row=n, col=1)

        # ── Event highlighting ───────────────────────────────────────────
        if event_id_col in df.columns and (
            train_event_ids is not None or val_event_ids is not None
        ):
            spans = (
                df.groupby(event_id_col)[datetime_col]
                .agg(["min", "max"])
                .to_dict(orient="index")
            )

            for eid in (train_event_ids or []):
                if eid in spans:
                    fig.add_vrect(
                        x0=spans[eid]["min"],
                        x1=spans[eid]["max"],
                        fillcolor="rgba(0,120,215,0.10)",
                        line_width=0,
                        row="all",
                        col="all",
                    )
            for eid in (val_event_ids or []):
                if eid in spans:
                    fig.add_vrect(
                        x0=spans[eid]["min"],
                        x1=spans[eid]["max"],
                        fillcolor="rgba(230,115,0,0.10)",
                        line_width=0,
                        row="all",
                        col="all",
                    )

            # Legend entries for train / validation bands
            if train_event_ids:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        name="Training",
                        line=dict(color="rgba(0,120,215,0.7)", width=10),
                        showlegend=True,
                    ),
                    row=1, col=1,
                )
            if val_event_ids:
                fig.add_trace(
                    go.Scatter(
                        x=[None], y=[None], mode="lines",
                        name="Validation",
                        line=dict(color="rgba(230,115,0,0.7)", width=10),
                        showlegend=True,
                    ),
                    row=1, col=1,
                )

        fig.update_layout(
            title=title,
            height=max(300 * n, 400),
            hovermode="x unified",
            legend=dict(x=1.02, y=1.0),
            barmode="overlay",
        )
        return fig

    # ------------------------------------------------------------------
    # plot_ensemble_timeseries
    # ------------------------------------------------------------------

    def plot_ensemble_timeseries(
        datetime,
        rainfall_mm,
        observed: np.ndarray,
        pred_df: pd.DataFrame,
        aliases: list,
        output_name: str = "ensemble_output",
        observed_label: str = "Observed",
        rainfall_label: str = "Rainfall [mm]",
        title: str = "Ensemble Components",
    ) -> "go.Figure":
        """Two-panel ensemble component breakdown.

        **Row 1 (25%)** — Rainfall bars with an inverted Y-axis.  Omitted
        entirely when *rainfall_mm* is ``None``.

        **Row 2 (75%)** — Observed flow (black solid line), total predicted
        (crimson dashed line), and one **stacked area** per alias showing each
        component's individual contribution.

        :param datetime: Time axis aligned with *observed*.
        :param rainfall_mm: Rainfall depth array, or ``None`` to skip the top
            panel.
        :param observed: 1-D array of observed flow values.
        :param pred_df: Output of :meth:`~sparsehydro.ensemble.EnsembleModel.predict`
            containing ``{alias}_output`` columns and the *output_name* column.
        :param aliases: List of alias strings matching the EnsembleModel's
            ``aliases`` attribute (e.g. ``["rdii", "sanitary"]``).
        :param output_name: Column name for the combined output in *pred_df*.
        :param observed_label: Legend label for the observed trace.
        :param rainfall_label: Y-axis label and legend label for rainfall.
        :param title: Figure title.
        :returns: Plotly Figure with 1–2 rows.
        :rtype: plotly.graph_objects.Figure
        """
        observed = np.asarray(observed, dtype=float)
        has_rain = rainfall_mm is not None

        component_colors = [
            ("rgba(31,119,180,0.45)",  "#1f77b4"),
            ("rgba(44,160,44,0.45)",   "#2ca02c"),
            ("rgba(255,127,14,0.45)",  "#ff7f0e"),
            ("rgba(148,103,189,0.45)", "#9467bd"),
            ("rgba(214,39,40,0.45)",   "#d62728"),
        ]

        if has_rain:
            fig = make_subplots(
                rows=2, cols=1,
                shared_xaxes=True,
                row_heights=[0.25, 0.75],
                vertical_spacing=0.04,
                subplot_titles=["Rainfall", "Flow Components"],
            )
            fig.add_trace(
                go.Bar(
                    x=datetime,
                    y=np.asarray(rainfall_mm, dtype=float),
                    name=rainfall_label,
                    marker_color="steelblue",
                    opacity=0.7,
                ),
                row=1, col=1,
            )
            fig.update_yaxes(autorange="reversed", title_text=rainfall_label, row=1, col=1)
            flow_row = 2
        else:
            fig = make_subplots(rows=1, cols=1, subplot_titles=["Flow Components"])
            flow_row = 1

        pred_dt = pred_df["datetime"] if "datetime" in pred_df.columns else datetime

        # Stacked component areas (drawn first so they appear behind the lines)
        for i, alias in enumerate(aliases):
            col_name = f"{alias}_output"
            if col_name not in pred_df.columns:
                continue
            fill_color, line_color = component_colors[i % len(component_colors)]
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[col_name].to_numpy(dtype=float),
                    mode="lines",
                    name=alias.replace("_", " ").title(),
                    stackgroup="components",
                    fillcolor=fill_color,
                    line=dict(color=line_color, width=1),
                ),
                row=flow_row, col=1,
            )

        # Total predicted (dashed line, on top of stacked areas)
        if output_name in pred_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=pred_dt,
                    y=pred_df[output_name].to_numpy(dtype=float),
                    mode="lines",
                    name="Predicted (total)",
                    line=dict(color="crimson", width=2, dash="dash"),
                ),
                row=flow_row, col=1,
            )

        # Observed (solid line, drawn last to appear on top)
        fig.add_trace(
            go.Scatter(
                x=datetime,
                y=observed,
                mode="lines",
                name=observed_label,
                line=dict(color="black", width=1.5),
            ),
            row=flow_row, col=1,
        )

        fig.update_yaxes(title_text="Flow", row=flow_row, col=1)
        fig.update_xaxes(title_text="Date / Time", row=flow_row, col=1)
        fig.update_layout(
            title=title,
            height=600,
            hovermode="x unified",
            legend=dict(x=1.02, y=1.0),
        )
        return fig

except ImportError:

    def plot_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_timeseries. Install with: pip install plotly")

    def plot_residuals_scatter(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_residuals_scatter. Install with: pip install plotly")

    def plot_cumulative_volume(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_cumulative_volume. Install with: pip install plotly")

    def plot_calibration_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_calibration_timeseries. Install with: pip install plotly")

    def plot_data_explorer(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_data_explorer. Install with: pip install plotly")

    def plot_ensemble_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError("plotly is required for plot_ensemble_timeseries. Install with: pip install plotly")

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
