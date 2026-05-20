"""General-purpose interactive visualizations for any sparsehydro model.

Requires the optional ``plotly`` dependency::

    pip install plotly

Three public plot functions are provided:

- :func:`plot_timeseries` — dual-axis subplot (rainfall + observed/predicted).
- :func:`plot_pareto_evolution` — animated Pareto front with generation slider.
- :func:`plot_parallel_coordinates` — parallel-axis trade-off explorer.

All calibration-result plots accept any
:class:`~sparsehydro.calibration.result.CalibrationResult`, so they work with
any solver and any model.

:class:`VisualizationModel` wraps these functions behind the standard
:class:`~sparsehydro.interfaces.IModel` lifecycle and accepts any model's
output DataFrame by letting the caller specify which column contains which
signal::

    viz = VisualizationModel(title="My Model Run")
    viz.initialize()
    viz.validate()
    viz.prepare(
        result_df,
        datetime_col="datetime",
        predicted_col="rdii_mm",
        observed_col="flow_cfs",
        rainfall_col="rainfall_mm",
    )
    viz.predict()
    viz.timeseries_figure.show()
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from .enums import ModelState
from .interfaces import IModel

if TYPE_CHECKING:
    from .calibration.result import CalibrationResult

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


# ======================================================================
# Pure plot functions
# ======================================================================

try:
    import plotly.graph_objects as go  # type: ignore[import]
    from plotly.subplots import make_subplots  # type: ignore[import]

    # ------------------------------------------------------------------
    # Time series
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
    # Pareto front evolution with generation slider
    # ------------------------------------------------------------------

    def plot_pareto_evolution(
        result: "CalibrationResult",
        x_obj: Union[int, str] = 0,
        y_obj: Union[int, str] = 1,
        title: str = "NSGA-II Pareto Front Evolution",
    ) -> go.Figure:
        """Animated scatter showing all solutions and the Pareto front per generation.

        A slider at the bottom navigates between generations.  A Play/Pause
        button animates the evolution automatically.

        Each frame shows:

        - Grey markers for every solution in that generation.
        - Viridis-coloured markers (coloured by Y-objective value) for the
          Pareto-front subset.

        :param result: Calibration result with per-generation history.
        :type result: CalibrationResult
        :param x_obj: X-axis objective — zero-based index or objective name.
            Defaults to ``0`` (first objective).
        :param y_obj: Y-axis objective — zero-based index or objective name.
            Defaults to ``1`` (second objective).
        :param title: Figure title.
        :returns: Plotly Figure with animation frames and slider.
        :rtype: plotly.graph_objects.Figure
        """
        from .calibration.result import _identify_pareto

        obj_names = result.objective_names
        min_flags = result.minimize_flags

        def _resolve(ref: Union[int, str]) -> int:
            if isinstance(ref, str):
                if ref not in obj_names:
                    raise ValueError(
                        f"Objective {ref!r} not found. Available: {obj_names}"
                    )
                return obj_names.index(ref)
            return int(ref)

        xi = _resolve(x_obj)
        yi = _resolve(y_obj)
        x_label = obj_names[xi]
        y_label = obj_names[yi]

        def _display(F: np.ndarray, col: int) -> np.ndarray:
            vals = F[:, col]
            return vals if min_flags[col] else -vals

        frames = []
        slider_steps = []

        for rec in result.history:
            all_x = _display(rec.F, xi)
            all_y = _display(rec.F, yi)
            is_pareto = _identify_pareto(rec.F)
            par_x = all_x[is_pareto]
            par_y = all_y[is_pareto]

            frame_data = [
                go.Scatter(
                    x=all_x,
                    y=all_y,
                    mode="markers",
                    marker=dict(color="lightgrey", size=5),
                    name="All solutions",
                    showlegend=False,
                ),
                go.Scatter(
                    x=par_x,
                    y=par_y,
                    mode="markers",
                    marker=dict(
                        color=par_y,
                        colorscale="Viridis",
                        size=9,
                        showscale=True,
                        colorbar=dict(title=y_label, x=1.02),
                        line=dict(color="black", width=0.5),
                    ),
                    name="Pareto front",
                    showlegend=False,
                ),
            ]
            frames.append(go.Frame(data=frame_data, name=str(rec.generation)))
            slider_steps.append(
                dict(
                    args=[
                        [str(rec.generation)],
                        {
                            "frame": {"duration": 0, "redraw": True},
                            "mode": "immediate",
                            "transition": {"duration": 0},
                        },
                    ],
                    label=str(rec.generation),
                    method="animate",
                )
            )

        first = result.history[0]
        is_pareto0 = _identify_pareto(first.F)
        init_x = _display(first.F, xi)
        init_y = _display(first.F, yi)
        init_data = [
            go.Scatter(
                x=init_x,
                y=init_y,
                mode="markers",
                marker=dict(color="lightgrey", size=5),
                name="All solutions",
            ),
            go.Scatter(
                x=init_x[is_pareto0],
                y=init_y[is_pareto0],
                mode="markers",
                marker=dict(
                    color=init_y[is_pareto0],
                    colorscale="Viridis",
                    size=9,
                    showscale=True,
                    colorbar=dict(title=y_label, x=1.02),
                    line=dict(color="black", width=0.5),
                ),
                name="Pareto front",
            ),
        ]

        layout = go.Layout(
            title=title,
            xaxis=dict(title=x_label),
            yaxis=dict(title=y_label),
            height=550,
            updatemenus=[
                dict(
                    type="buttons",
                    showactive=False,
                    x=0.0,
                    y=-0.12,
                    xanchor="left",
                    yanchor="top",
                    buttons=[
                        dict(
                            label="▶ Play",
                            method="animate",
                            args=[
                                None,
                                {
                                    "frame": {"duration": 150, "redraw": True},
                                    "fromcurrent": True,
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                        dict(
                            label="⏸ Pause",
                            method="animate",
                            args=[
                                [None],
                                {
                                    "frame": {"duration": 0, "redraw": False},
                                    "mode": "immediate",
                                    "transition": {"duration": 0},
                                },
                            ],
                        ),
                    ],
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=slider_steps,
                    x=0.0,
                    len=1.0,
                    y=-0.05,
                    currentvalue=dict(
                        prefix="Generation: ",
                        visible=True,
                        xanchor="center",
                    ),
                    transition=dict(duration=0),
                )
            ],
        )

        return go.Figure(data=init_data, frames=frames, layout=layout)

    # ------------------------------------------------------------------
    # Parallel coordinates
    # ------------------------------------------------------------------

    def plot_parallel_coordinates(
        result: "CalibrationResult",
        color_by: Union[str, None] = None,
        title: str = "Pareto Front — Parallel Coordinates",
        use_final_pareto_only: bool = True,
    ) -> go.Figure:
        """Parallel-axis trade-off explorer for Pareto solutions.

        Each line represents one solution.  Axes are the calibrated parameters
        followed by each objective (in display form).  Drag the endpoints of
        any axis to filter solutions (Plotly ``constraintrange`` built-in).

        :param result: Calibration result.
        :type result: CalibrationResult
        :param color_by: Objective name to use for line colour.  Defaults to
            the first objective.
        :param title: Figure title.
        :param use_final_pareto_only: Plot only the final Pareto front when
            ``True``; plot all final-generation solutions when ``False``.
        :returns: Plotly Figure with ``go.Parcoords`` trace.
        :rtype: plotly.graph_objects.Figure
        """
        df = result.to_pareto_dataframe()
        if use_final_pareto_only:
            plot_df = df[df["is_pareto"]].reset_index(drop=True)
        else:
            final_gen = df["generation"].max()
            plot_df = df[df["generation"] == final_gen].reset_index(drop=True)

        if len(plot_df) == 0:
            final_gen = df["generation"].max()
            plot_df = df[df["generation"] == final_gen].reset_index(drop=True)

        obj_names = result.objective_names
        color_col = color_by if (color_by and color_by in obj_names) else obj_names[0]

        def _dim(col: str) -> dict:
            vals = plot_df[col].tolist()
            lo, hi = min(vals), max(vals)
            if lo == hi:
                lo -= 1e-6
                hi += 1e-6
            return dict(label=col, values=vals, range=[lo, hi])

        dims = [_dim(p) for p in result.param_names]
        dims += [_dim(o) for o in obj_names]

        color_vals = plot_df[color_col].tolist()
        fig = go.Figure(
            data=go.Parcoords(
                line=dict(
                    color=color_vals,
                    colorscale="Viridis",
                    showscale=True,
                    colorbar=dict(title=color_col),
                    cmin=min(color_vals),
                    cmax=max(color_vals),
                ),
                dimensions=dims,
            )
        )
        fig.update_layout(title=title, height=500)
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
            self._calibration_result: CalibrationResult | None = None
            self._timeseries_fig: go.Figure | None = None
            self._pareto_fig: go.Figure | None = None
            self._parallel_fig: go.Figure | None = None

        # --------------------------------------------------------------
        # IModel lifecycle
        # --------------------------------------------------------------

        def initialize(self) -> None:
            """Advance to INITIALIZED.

            Visualization has no calibratable numerical parameters.
            """
            self._state = ModelState.INITIALIZED

        def validate(self) -> bool:
            """Validate and advance to VALIDATED.

            :returns: Always ``True``.
            :rtype: bool
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
            """Load model output data and configure which column contains which signal.

            :param data: Any model's output DataFrame.
            :type data: pandas.DataFrame
            :param datetime_col: Column name for the time axis.
            :param predicted_col: Column name for the predicted output signal.
            :param observed_col: Column name for observed data, or ``None`` to omit.
            :param rainfall_col: Column name for rainfall forcing, or ``None`` to omit.
            :param calibration_result: When provided, :meth:`predict` also
                generates the Pareto-evolution and parallel-coordinates figures.
            :type calibration_result: CalibrationResult, optional
            :raises ValueError: If required columns are absent.
            """
            missing = []
            for col in (datetime_col, predicted_col):
                if col not in data.columns:
                    missing.append(col)
            if missing:
                raise ValueError(
                    f"prepare() data is missing required columns: {missing}"
                )
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
            :rtype: pandas.DataFrame
            :raises RuntimeError: If :meth:`prepare` has not been called.
            """
            if self._data is None:
                raise RuntimeError("Call prepare(data) before predict().")

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

        # --------------------------------------------------------------
        # Figure properties
        # --------------------------------------------------------------

        @property
        def timeseries_figure(self) -> go.Figure | None:
            """Dual-axis rainfall/flow time series, or ``None`` before :meth:`predict`.

            :rtype: plotly.graph_objects.Figure or None
            """
            return self._timeseries_fig

        @property
        def pareto_figure(self) -> go.Figure | None:
            """Animated Pareto-front evolution, or ``None`` when no
            ``CalibrationResult`` was supplied or before :meth:`predict`.

            :rtype: plotly.graph_objects.Figure or None
            """
            return self._pareto_fig

        @property
        def parallel_figure(self) -> go.Figure | None:
            """Parallel-coordinates trade-off explorer, or ``None`` when no
            ``CalibrationResult`` was supplied or before :meth:`predict`.

            :rtype: plotly.graph_objects.Figure or None
            """
            return self._parallel_fig

except ImportError:

    def plot_timeseries(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_timeseries. "
            "Install with: pip install plotly"
        )

    def plot_pareto_evolution(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_pareto_evolution. "
            "Install with: pip install plotly"
        )

    def plot_parallel_coordinates(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_parallel_coordinates. "
            "Install with: pip install plotly"
        )

    class VisualizationModel(IModel):  # type: ignore[no-redef]
        model_name = "visualization"

        def initialize(self) -> None:
            raise ImportError(
                "plotly is required for VisualizationModel. "
                "Install with: pip install plotly"
            )

        def validate(self) -> bool:
            raise ImportError(
                "plotly is required for VisualizationModel. "
                "Install with: pip install plotly"
            )

        def prepare(self, data: pd.DataFrame, **kwargs) -> None:  # type: ignore[override]
            raise ImportError(
                "plotly is required for VisualizationModel. "
                "Install with: pip install plotly"
            )

        def predict(self) -> pd.DataFrame:
            raise ImportError(
                "plotly is required for VisualizationModel. "
                "Install with: pip install plotly"
            )

        def finalize(self) -> None:
            raise ImportError(
                "plotly is required for VisualizationModel. "
                "Install with: pip install plotly"
            )
