"""Multi-panel calibration dashboard — pure Plotly HTML, no server required.

Single public function:

- :func:`plot_calibration_dashboard` — combine multiple sub-figures into a
  single offline HTML file.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Union

if TYPE_CHECKING:
    import pandas as pd

    from ..calibration.result import CalibrationResult

    try:
        import plotly.graph_objects as go
    except ImportError:  # pragma: no cover
        pass


try:
    import plotly.graph_objects as go  # type: ignore[import]
    import plotly.io as pio  # type: ignore[import]

    def plot_calibration_dashboard(
        result: "CalibrationResult",
        timeseries_df: "pd.DataFrame | None" = None,
        datetime_col: str = "datetime",
        predicted_col: str = "predicted",
        observed_col: Union[str, None] = None,
        rainfall_col: Union[str, None] = None,
        flow_label: str = "Flow",
        title: str = "Calibration Dashboard",
        output_path: Union[str, None] = None,
        use_cdn: bool = True,
    ) -> "go.Figure":
        """Build a multi-panel calibration dashboard and write it to an HTML file.

        Because :class:`plotly.graph_objects.Frame` objects (used by
        :func:`plot_pareto_evolution`) and ``go.Parcoords`` traces cannot be
        embedded in a shared ``make_subplots`` grid, each sub-figure is
        serialised individually with ``plotly.io.to_html(full_html=False)``
        and the sections are joined in an HTML template with a single Plotly
        CDN ``<script>`` tag.

        **Panels** (shown when data is available):

        1. Time series — if *timeseries_df* is supplied.
        2. Objective convergence — always shown.
        3. Pareto front evolution (animated) — requires ≥ 1 history record.
        4. Parameter distributions — always shown.
        5. Parallel coordinates — always shown.

        :param result: Calibration result.
        :type result: CalibrationResult
        :param timeseries_df: Optional model-output DataFrame.
        :type timeseries_df: pandas.DataFrame or None
        :param datetime_col: Datetime column name in *timeseries_df*.
        :type datetime_col: str
        :param predicted_col: Predicted-flow column name in *timeseries_df*.
        :type predicted_col: str
        :param observed_col: Observed-flow column name, or ``None`` to omit.
        :type observed_col: str or None
        :param rainfall_col: Rainfall column name, or ``None`` to omit zeros.
        :type rainfall_col: str or None
        :param flow_label: Y-axis label for flow panels.
        :type flow_label: str
        :param title: Dashboard heading.
        :type title: str
        :param output_path: File path for the HTML output.  Pass ``None`` to
            skip writing.
        :type output_path: str or None
        :param use_cdn: Include Plotly via CDN ``<script>`` tag when ``True``;
            inline the full Plotly bundle when ``False`` (larger file, fully
            offline).
        :type use_cdn: bool
        :returns: The objective-convergence figure (for interactive notebook
            use; the full dashboard is only available in the HTML file).
        :rtype: plotly.graph_objects.Figure
        """
        import numpy as np

        from .calibration import (
            plot_objective_convergence,
            plot_parallel_coordinates,
            plot_parameter_distributions,
            plot_pareto_evolution,
        )
        from .timeseries import plot_timeseries

        panels: list[tuple[str, go.Figure]] = []

        # Panel 1 — time series
        if timeseries_df is not None:
            dt = timeseries_df[datetime_col]
            pred = timeseries_df[predicted_col].to_numpy(dtype=float)
            obs = (
                timeseries_df[observed_col].to_numpy(dtype=float)
                if observed_col and observed_col in timeseries_df.columns
                else None
            )
            rain = (
                timeseries_df[rainfall_col].to_numpy(dtype=float)
                if rainfall_col and rainfall_col in timeseries_df.columns
                else np.zeros(len(timeseries_df))
            )
            ts_fig = plot_timeseries(
                datetime=dt,
                rainfall_mm=rain,
                observed_flow=obs,
                predicted_flow=pred,
                title="Time Series",
                flow_label=flow_label,
            )
            panels.append(("Time Series", ts_fig))

        # Panel 2 — objective convergence
        conv_fig = plot_objective_convergence(result, title="Objective Convergence")
        panels.append(("Objective Convergence", conv_fig))

        # Panel 3 — animated Pareto evolution
        if result.history:
            pareto_fig = plot_pareto_evolution(result, title="Pareto Front Evolution")
            panels.append(("Pareto Front Evolution", pareto_fig))

        # Panel 4 — parameter distributions
        dist_fig = plot_parameter_distributions(result, title="Parameter Distributions")
        panels.append(("Parameter Distributions", dist_fig))

        # Panel 5 — parallel coordinates
        if result.pareto_X.shape[0] >= 2:
            pc_fig = plot_parallel_coordinates(result, title="Parallel Coordinates")
            panels.append(("Parallel Coordinates", pc_fig))

        # Build HTML
        if output_path is not None:
            if use_cdn:
                plotlyjs_tag = (
                    '<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>'
                )
                include_plotlyjs = False
            else:
                include_plotlyjs = True
                plotlyjs_tag = ""

            panel_htmls = []
            for panel_title, fig in panels:
                fig_html = pio.to_html(fig, full_html=False, include_plotlyjs=include_plotlyjs)
                if include_plotlyjs:
                    include_plotlyjs = False  # only embed once
                panel_htmls.append(
                    f'<section style="margin-bottom:2em;">'
                    f'<h2 style="font-family:sans-serif;color:#333;">{panel_title}</h2>'
                    f"{fig_html}"
                    f"</section>"
                )

            body = "\n".join(panel_htmls)
            html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
{plotlyjs_tag}
<style>
  body {{ font-family: sans-serif; margin: 1.5em; background: #fafafa; }}
  h1 {{ color: #222; border-bottom: 2px solid #aaa; padding-bottom: 0.3em; }}
  section {{ background: white; padding: 1em; border-radius: 6px;
             box-shadow: 0 1px 4px rgba(0,0,0,.1); }}
</style>
</head>
<body>
<h1>{title}</h1>
{body}
</body>
</html>"""
            pathlib.Path(output_path).write_text(html, encoding="utf-8")

        return conv_fig

except ImportError:

    def plot_calibration_dashboard(*args, **kwargs):  # type: ignore[misc]
        raise ImportError(
            "plotly is required for plot_calibration_dashboard. "
            "Install with: pip install plotly"
        )
