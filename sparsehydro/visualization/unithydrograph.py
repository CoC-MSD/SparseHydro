"""Plotly visualizations specific to unit hydrograph analysis.

Extends :mod:`sparsehydro.visualization` with six figure builders for the
sequential UH fitting workflow.  All functions return :class:`plotly.graph_objects.Figure`.

Functions
---------
- :func:`plot_rainfall_flow_with_events` — two-panel timeseries with event bands
- :func:`plot_filter_signals` — sg_0 / sg_1 / sg_2 with threshold lines
- :func:`plot_event_detection` — sg_0 + peak markers + event shading
- :func:`plot_sequential_fit` — observed vs global+per-event predicted + residual
- :func:`plot_parameter_evolution` — fitted params over time colored by NSE
- :func:`plot_effective_area` — bar chart of Ae per event
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ..events import EventRecord
from ..filters import FilterResult
from ..unithydrograph.sequential import SequentialFitSummary

_D3 = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def _event_color(idx: int, alpha: float = 0.15) -> str:
    hex_col = _D3[idx % len(_D3)].lstrip("#")
    r, g, b = int(hex_col[0:2], 16), int(hex_col[2:4], 16), int(hex_col[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


def _solid_event_color(idx: int) -> str:
    return _D3[idx % len(_D3)]


def _add_event_bands(fig, events, rows, show_label=True, label_row=1, opacity=0.13):
    for i, event in enumerate(events):
        color = _event_color(i, opacity)
        border = _solid_event_color(i)
        x_s, x_e = event.start_datetime.isoformat(), event.end_datetime.isoformat()
        for row in rows:
            fig.add_vrect(x0=x_s, x1=x_e, fillcolor=color,
                          line={"color": border, "width": 0.5, "dash": "dot"},
                          opacity=1.0, layer="below", row=row, col=1)
        if show_label:
            fig.add_annotation(
                x=event.peak_datetime.isoformat(), y=1.0, yref="paper",
                text=f"E{event.event_id}", showarrow=False,
                font={"size": 9, "color": border}, textangle=-90,
                xanchor="center", yanchor="top", row=label_row, col=1,
            )


def plot_rainfall_flow_with_events(
    rain_stormflow: pd.DataFrame,
    events: list[EventRecord],
    title: str = "Rainfall and Stormflow with Detected Events",
    rain_label: str = "Rainfall (mm/hr)",
    flow_label: str = "Stormflow (cfs)",
) -> go.Figure:
    """Two-panel timeseries: top = inverted rainfall bars, bottom = stormflow.

    Transparent colored bands highlight each event on both panels.
    """
    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.28, 0.72], vertical_spacing=0.04,
                        subplot_titles=[rain_label, flow_label])

    fig.add_trace(go.Bar(x=df["datetime"], y=df["rain"].values, name="Rainfall",
                         marker_color="steelblue", marker_opacity=0.7), row=1, col=1)
    fig.add_trace(go.Scatter(x=df["datetime"], y=df["stormflow"].values, name="Stormflow",
                             mode="lines", line={"color": "black", "width": 1.5}), row=2, col=1)

    _add_event_bands(fig, events, rows=[1, 2], show_label=True, label_row=1)

    if events:
        ae_x = [e.peak_datetime for e in events]
        ae_y = [float(df.loc[df["datetime"].sub(e.peak_datetime).abs().idxmin(), "stormflow"]) for e in events]
        ae_text = [f"Event {e.event_id}<br>Ae={e.effective_area:.3f}<br>"
                   f"{e.start_datetime.date()} – {e.end_datetime.date()}" for e in events]
        fig.add_trace(go.Scatter(x=ae_x, y=ae_y, mode="markers",
                                 marker={"size": 8, "color": "crimson", "symbol": "diamond"},
                                 name="Event peak", text=ae_text, hoverinfo="text"), row=2, col=1)

    fig.update_yaxes(title_text=rain_label, autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text=flow_label, row=2, col=1)
    fig.update_xaxes(title_text="Date", row=2, col=1,
                     rangeslider={"visible": True, "thickness": 0.05})
    fig.update_layout(title={"text": title, "x": 0.5}, hovermode="x unified",
                      legend={"orientation": "h", "y": -0.15}, height=600)
    return fig


def plot_filter_signals(
    filter_result: FilterResult,
    title: str = "Savitzky-Golay Filter Signals",
) -> go.Figure:
    """Three-panel figure: sg_0, sg_1, sg_2 with threshold lines."""
    t = filter_result.datetime
    ths = filter_result.thresholds
    sg_0_th = ths.get("sg_0_th", 0.0)
    sg_1_th = ths.get("sg_1_th", 0.0)
    sg_2_th = ths.get("sg_2_th", 0.0)

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.4, 0.3, 0.3], vertical_spacing=0.05,
                        subplot_titles=["sg_0 (smoothed flow)", "sg_1 (slope)", "sg_2 (curvature)"])

    fig.add_trace(go.Scatter(x=t, y=filter_result.raw_flow, mode="lines",
                             line={"color": "lightgrey", "width": 1}, name="Raw flow"), row=1, col=1)
    fig.add_trace(go.Scatter(x=t, y=filter_result.sg_0, mode="lines",
                             line={"color": "#1f77b4", "width": 1.5}, name="sg_0"), row=1, col=1)
    if sg_0_th > 0:
        fig.add_hline(y=sg_0_th, line={"color": "red", "dash": "dash", "width": 1.2},
                      annotation_text="sg_0_th", annotation_position="right", row=1, col=1)

    fig.add_trace(go.Scatter(x=t, y=filter_result.sg_1, mode="lines",
                             line={"color": "#ff7f0e", "width": 1.5}, name="sg_1"), row=2, col=1)
    fig.add_hline(y=0, line={"color": "grey", "dash": "solid", "width": 0.8}, row=2, col=1)
    if sg_1_th > 0:
        fig.add_hline(y=sg_1_th, line={"color": "red", "dash": "dash", "width": 1},
                      annotation_text="+sg_1_th", annotation_position="right", row=2, col=1)
        fig.add_hline(y=-sg_1_th, line={"color": "red", "dash": "dash", "width": 1},
                      annotation_text="-sg_1_th", annotation_position="right", row=2, col=1)

    fig.add_trace(go.Scatter(x=t, y=filter_result.sg_2, mode="lines",
                             line={"color": "#2ca02c", "width": 1.5}, name="sg_2"), row=3, col=1)
    fig.add_hline(y=0, line={"color": "grey", "dash": "solid", "width": 0.8}, row=3, col=1)
    if sg_2_th > 0:
        fig.add_hline(y=sg_2_th, line={"color": "red", "dash": "dash", "width": 1},
                      annotation_text="+sg_2_th", annotation_position="right", row=3, col=1)
        fig.add_hline(y=-sg_2_th, line={"color": "red", "dash": "dash", "width": 1},
                      annotation_text="-sg_2_th", annotation_position="right", row=3, col=1)

    fig.update_layout(title={"text": title, "x": 0.5}, hovermode="x unified",
                      height=650, legend={"orientation": "h", "y": -0.08})
    return fig


def plot_event_detection(
    filter_result: FilterResult,
    events: list[EventRecord],
    title: str = "Event Detection — sg_0 with Detected Events",
) -> go.Figure:
    """Single panel: sg_0 + peak markers + event shading."""
    t = filter_result.datetime
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=t, y=filter_result.sg_0, mode="lines",
                             line={"color": "#1f77b4", "width": 1.5}, name="sg_0"))

    ths = filter_result.thresholds
    if ths.get("sg_0_th", 0.0) > 0:
        fig.add_hline(y=ths["sg_0_th"], line={"color": "red", "dash": "dash", "width": 1.2},
                      annotation_text="sg_0_th")

    if events:
        dt_arr = pd.to_datetime(filter_result.datetime)
        sg0_arr = filter_result.sg_0
        peak_x = [e.peak_datetime for e in events]
        peak_y = [float(sg0_arr[int(dt_arr.sub(e.peak_datetime).abs().argmin())]) for e in events]
        peak_text = [f"Event {e.event_id}<br>sg_0={pf:.3f}<br>Ae={e.effective_area:.3f}"
                     for e, pf in zip(events, peak_y)]
        fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers",
                                 marker={"size": 10, "color": "crimson", "symbol": "diamond"},
                                 name="Peaks", text=peak_text, hoverinfo="text"))
        for i, event in enumerate(events):
            color = _solid_event_color(i)
            fig.add_vline(x=event.start_datetime.isoformat(),
                          line={"color": color, "width": 1, "dash": "longdash"})
            fig.add_vline(x=event.end_datetime.isoformat(),
                          line={"color": color, "width": 1, "dash": "dot"})
            fig.add_vrect(x0=event.start_datetime.isoformat(), x1=event.end_datetime.isoformat(),
                          fillcolor=_event_color(i, 0.12),
                          line={"color": color, "width": 0.5}, opacity=1.0, layer="below")

    fig.update_layout(title={"text": title, "x": 0.5}, xaxis_title="Date",
                      yaxis_title="sg_0", hovermode="x unified", height=450)
    return fig


def plot_sequential_fit(
    summary: SequentialFitSummary,
    rain_stormflow: pd.DataFrame,
    events: list[EventRecord] | None = None,
    title: str = "Sequential Unit Hydrograph Fitting",
) -> go.Figure:
    """Three-row panel: rainfall / observed+predicted / residual."""
    if events is None:
        events = summary.events

    df = rain_stormflow.copy()
    df["datetime"] = pd.to_datetime(df["datetime"])
    global_pred = summary.global_predicted.copy()
    global_pred["datetime"] = pd.to_datetime(global_pred["datetime"])

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        row_heights=[0.20, 0.50, 0.30], vertical_spacing=0.04,
                        subplot_titles=["Rainfall", "Observed vs Predicted Flow",
                                        "Residual (Obs − Pred)"])

    fig.add_trace(go.Bar(x=df["datetime"], y=df["rain"].values, name="Rainfall",
                         marker_color="steelblue", marker_opacity=0.7), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    obs_df = summary.global_observed.copy()
    obs_df["datetime"] = pd.to_datetime(obs_df["datetime"])
    fig.add_trace(go.Scatter(x=obs_df["datetime"], y=obs_df["stormflow"], mode="lines",
                             line={"color": "black", "width": 1.5}, name="Observed"), row=2, col=1)
    fig.add_trace(go.Scatter(x=global_pred["datetime"], y=global_pred["Q_pred"], mode="lines",
                             line={"color": "crimson", "width": 1.5, "dash": "dash"},
                             name="Global predicted"), row=2, col=1)

    for i, (event, cal_result) in enumerate(zip(summary.events, summary.calibration_results)):
        color = _solid_event_color(i)
        event_mask = (global_pred["datetime"] >= event.start_datetime) & (global_pred["datetime"] <= event.end_datetime)
        ep_t = global_pred.loc[event_mask, "datetime"]
        ep_v = global_pred.loc[event_mask, "Q_pred"]
        fig.add_trace(go.Scatter(x=ep_t, y=ep_v, mode="lines",
                                 line={"color": color, "width": 2},
                                 name=f"E{event.event_id} pred", showlegend=(i < 10)), row=2, col=1)

        disp = cal_result.objective_display_values()[0]
        obj_names = cal_result.objective_names
        rmse_idx = next((j for j, n in enumerate(obj_names) if "rmse" in n.lower()), None)
        nse_idx = next((j for j, n in enumerate(obj_names) if "nash" in n.lower()), None)
        ann_lines = []
        if rmse_idx is not None:
            ann_lines.append(f"RMSE={disp[rmse_idx]:.2f}")
        if nse_idx is not None:
            ann_lines.append(f"NSE={disp[nse_idx]:.2f}")
        if ann_lines:
            peak_mask = obs_df["datetime"].sub(event.peak_datetime).abs() == obs_df["datetime"].sub(event.peak_datetime).abs().min()
            peak_y = float(obs_df.loc[peak_mask, "stormflow"].iloc[0]) if peak_mask.any() else 0.0
            fig.add_annotation(x=event.peak_datetime.isoformat(), y=peak_y,
                                text="<br>".join(ann_lines), showarrow=True, arrowhead=2,
                                arrowcolor=color, font={"size": 9, "color": color},
                                bgcolor="white", opacity=0.85, row=2, col=1)

    try:
        merged = pd.merge_asof(obs_df.sort_values("datetime"),
                               global_pred.sort_values("datetime"), on="datetime",
                               direction="nearest", tolerance=pd.Timedelta("1h"))
        if not merged.empty:
            r_vals = merged["stormflow"].values - merged["Q_pred"].values
            t_res = merged["datetime"]
            pos = np.where(r_vals >= 0, r_vals, 0.0)
            neg = np.where(r_vals < 0, r_vals, 0.0)
            fig.add_trace(go.Bar(x=t_res, y=pos, name="Residual (+)",
                                 marker_color="rgba(255,100,100,0.6)"), row=3, col=1)
            fig.add_trace(go.Bar(x=t_res, y=neg, name="Residual (−)",
                                 marker_color="rgba(100,100,255,0.6)"), row=3, col=1)
            fig.add_hline(y=0, line={"color": "grey", "width": 0.8}, row=3, col=1)
    except Exception:
        pass

    _add_event_bands(fig, events, rows=[2, 3], show_label=True, label_row=2)

    fig.update_yaxes(title_text="Rainfall", row=1, col=1)
    fig.update_yaxes(title_text="Flow", row=2, col=1)
    fig.update_yaxes(title_text="Residual", row=3, col=1)
    fig.update_xaxes(title_text="Date", row=3, col=1,
                     rangeslider={"visible": True, "thickness": 0.05})
    fig.update_layout(title={"text": title, "x": 0.5}, hovermode="x unified",
                      barmode="overlay", height=750,
                      legend={"orientation": "h", "y": -0.12})
    return fig


def plot_parameter_evolution(
    summary: SequentialFitSummary,
    title: str = "Parameter Evolution Across Fitted Events",
) -> go.Figure:
    """One subplot per shape parameter, colored by NSE with rolling mean."""
    if not summary.calibration_results:
        return go.Figure().update_layout(title=title)

    param_evo = summary.parameter_evolution()
    metrics = summary.metrics_summary()

    all_param_cols = [c for c in param_evo.columns
                      if c not in ("event_id", "start_datetime", "end_datetime")]
    shape_params = [p for p in all_param_cols
                    if not (p == "A" or p.endswith("_A") or p.startswith("A_"))]
    if not shape_params:
        shape_params = all_param_cols
    if not shape_params:
        return go.Figure().update_layout(title=title)

    nse_col = next((c for c in metrics.columns if "nash" in c.lower() or "nse" in c.lower()), None)
    nse_vals = metrics[nse_col].values.astype(float) if nse_col else np.zeros(len(param_evo))
    nse_color = np.clip(nse_vals, -1, 1)
    dates = pd.to_datetime(param_evo["start_datetime"])

    n_params = len(shape_params)
    fig = make_subplots(rows=1, cols=n_params, subplot_titles=shape_params, horizontal_spacing=0.08)

    for j, pname in enumerate(shape_params):
        vals = param_evo[pname].values.astype(float)
        roll_mean = pd.Series(vals).rolling(window=5, min_periods=1, center=True).mean().values
        roll_std = pd.Series(vals).rolling(window=5, min_periods=1, center=True).std().fillna(0).values

        fig.add_trace(go.Scatter(
            x=list(dates) + list(dates[::-1]),
            y=list(roll_mean + roll_std) + list((roll_mean - roll_std)[::-1]),
            fill="toself", fillcolor="rgba(150,150,150,0.2)",
            line={"color": "rgba(0,0,0,0)"}, name="±1 std" if j == 0 else None,
            showlegend=(j == 0), hoverinfo="skip",
        ), row=1, col=j + 1)

        fig.add_trace(go.Scatter(
            x=dates, y=vals, mode="markers",
            marker={"size": 8, "color": nse_color, "colorscale": "RdYlGn",
                    "cmin": -1, "cmax": 1, "showscale": (j == n_params - 1),
                    "colorbar": {"title": "NSE", "thickness": 12} if j == n_params - 1 else {}},
            text=[f"Event {eid}<br>{pname}={v:.3f}<br>NSE={nse:.3f}"
                  for eid, v, nse in zip(param_evo["event_id"], vals, nse_vals)],
            hoverinfo="text", showlegend=False,
        ), row=1, col=j + 1)

        fig.add_trace(go.Scatter(
            x=dates, y=roll_mean, mode="lines",
            line={"color": "grey", "dash": "dash", "width": 1.5},
            name="5-event mean" if j == 0 else None, showlegend=(j == 0),
        ), row=1, col=j + 1)

        fig.update_yaxes(title_text=pname, row=1, col=j + 1)

    fig.update_layout(title={"text": title, "x": 0.5}, height=420, hovermode="closest")
    return fig


def plot_effective_area(
    events: list[EventRecord],
    title: str | None = None,
) -> go.Figure:
    """Bar chart of effective area (Ae) per event, colored by duration."""
    if not events:
        return go.Figure().update_layout(title=title or "Effective Area (no events)")

    ids = [e.event_id for e in events]
    ae = np.array([e.effective_area for e in events])
    durations = np.array([e.duration_hours() for e in events])
    mean_ae, std_ae = float(np.mean(ae)), float(np.std(ae))

    if title is None:
        title = f"Effective Area per Event — mean={mean_ae:.3f} ± {std_ae:.3f}"

    roll_mean = pd.Series(ae).rolling(window=5, min_periods=1, center=True).mean().values
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[f"E{i}" for i in ids], y=ae, name="Ae",
        marker={"color": durations, "colorscale": "Blues",
                "colorbar": {"title": "Duration (hr)", "thickness": 12}, "showscale": True},
        text=[f"Ae={v:.3f}<br>Dur={d:.1f}hr" for v, d in zip(ae, durations)], hoverinfo="text",
    ))
    fig.add_trace(go.Scatter(
        x=[f"E{i}" for i in ids], y=roll_mean, mode="lines+markers",
        line={"color": "crimson", "dash": "dash", "width": 2}, marker={"size": 6},
        name="5-event rolling mean",
    ))
    fig.add_hline(y=mean_ae, line={"color": "green", "dash": "longdash", "width": 1.5},
                  annotation_text=f"Mean={mean_ae:.3f}", annotation_position="right")
    fig.add_hrect(y0=mean_ae - std_ae, y1=mean_ae + std_ae,
                  fillcolor="rgba(0,200,0,0.08)", line_width=0, annotation_text="±1σ")

    fig.update_layout(title={"text": title, "x": 0.5}, xaxis_title="Event ID",
                      yaxis_title="Effective Area (Ae)", height=420,
                      legend={"orientation": "h", "y": -0.15})
    return fig


__all__ = [
    "plot_rainfall_flow_with_events",
    "plot_filter_signals",
    "plot_event_detection",
    "plot_sequential_fit",
    "plot_parameter_evolution",
    "plot_effective_area",
]
