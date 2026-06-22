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


def _nearest_idx(dt_ns: np.ndarray, target: pd.Timestamp) -> int:
    """Binary-search index of the element in *dt_ns* closest to *target*.

    *dt_ns* must be a sorted int64 array of nanoseconds-since-epoch values
    (obtained via ``pd.to_datetime(series).values.astype('int64')``).
    O(log N) vs the previous O(N) linear scan.
    """
    t = np.int64(target.value)
    i = int(np.searchsorted(dt_ns, t, side="left"))
    if i == 0:
        return 0
    if i >= len(dt_ns):
        return len(dt_ns) - 1
    return i if (dt_ns[i] - t) <= (t - dt_ns[i - 1]) else i - 1


def _downsample_bar(t: np.ndarray, y: np.ndarray, max_pts: int = 2000):
    """Reduce a large array to at most *max_pts* points for Bar rendering.

    Uses max-within-bin so rainfall peaks are preserved.
    Returns (t_ds, y_ds); unchanged if ``len(y) <= max_pts``.
    """
    n = len(y)
    if n <= max_pts:
        return t, y
    bin_size = max(1, n // max_pts)
    bin_starts = np.arange(0, n, bin_size)
    y_ds = np.maximum.reduceat(np.maximum(y, 0.0), bin_starts)
    t_ds = t[bin_starts]
    return t_ds, y_ds


def _row_yref(row: int) -> str:
    """Return the Plotly yref string for the given subplot row (1-indexed)."""
    return "y domain" if row == 1 else f"y{row} domain"


def _add_event_bands(fig, events, rows, show_label=True, label_row=1, opacity=0.13):
    """Add shaded bands and optional E-labels for each event.

    All shapes and annotations are added in a single ``update_layout`` call
    (batched) rather than one ``add_vrect`` / ``add_annotation`` call per event,
    which eliminates the per-call layout-mutation overhead.
    """
    if not events:
        return

    shapes = list(fig.layout.shapes or [])
    annotations = list(fig.layout.annotations or [])

    for i, event in enumerate(events):
        color = _event_color(i, opacity)
        border = _solid_event_color(i)
        x_s = event.start_datetime.isoformat()
        x_e = event.end_datetime.isoformat()

        for row in rows:
            shapes.append(dict(
                type="rect",
                xref="x", yref=_row_yref(row),
                x0=x_s, x1=x_e, y0=0, y1=1,
                fillcolor=color,
                line=dict(color=border, width=0.5, dash="dot"),
                layer="below",
            ))

        if show_label:
            # Use midpoint so labels work even when peak_datetime is NaT (CSV events)
            mid_dt = event.start_datetime + (event.end_datetime - event.start_datetime) / 2
            annotations.append(dict(
                x=mid_dt.isoformat(), y=1.0,
                xref="x", yref="paper",
                text=f"E{event.event_id}", showarrow=False,
                font=dict(size=9, color=border), textangle=-90,
                xanchor="center", yanchor="top",
            ))

    fig.update_layout(shapes=shapes, annotations=annotations)


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
    t = df["datetime"].values
    rain = df["rain"].values
    flow = df["stormflow"].values

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.28, 0.72], vertical_spacing=0.04,
                        subplot_titles=[rain_label, flow_label])

    # Downsample rainfall bars — preserves peaks, eliminates 100k-rect overhead
    t_rain, rain_ds = _downsample_bar(t, rain)
    fig.add_trace(go.Bar(x=t_rain, y=rain_ds, name="Rainfall",
                         marker_color="steelblue", marker_opacity=0.7), row=1, col=1)

    # Stormflow scatter — fast at any row count; no downsampling needed
    fig.add_trace(go.Scatter(x=t, y=flow, name="Stormflow",
                             mode="lines", line={"color": "black", "width": 1.5}), row=2, col=1)

    # Batch all event bands in one layout call
    _add_event_bands(fig, events, rows=[1, 2], show_label=True, label_row=1)

    if events:
        peak_x, peak_y = [], []
        for e in events:
            mask = (df["datetime"] >= e.start_datetime) & (df["datetime"] <= e.end_datetime)
            ev_flow = flow[mask]
            ev_t = t[mask]
            if len(ev_flow) > 0:
                idx = int(np.argmax(ev_flow))
                peak_x.append(pd.Timestamp(ev_t[idx]))
                peak_y.append(float(ev_flow[idx]))
            else:
                peak_x.append(e.start_datetime)
                peak_y.append(0.0)
        ae_text = [f"Event {e.event_id}<br>Ae={e.effective_area:.3f}<br>"
                   f"{e.start_datetime.date()} – {e.end_datetime.date()}" for e in events]
        fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers",
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
        dt_ns = pd.to_datetime(filter_result.datetime).values.astype("int64")
        sg0_arr = filter_result.sg_0
        peak_x = [e.peak_datetime for e in events]
        peak_y = [float(sg0_arr[_nearest_idx(dt_ns, e.peak_datetime)]) for e in events]
        peak_text = [f"Event {e.event_id}<br>sg_0={pf:.3f}<br>Ae={e.effective_area:.3f}"
                     for e, pf in zip(events, peak_y)]
        fig.add_trace(go.Scatter(x=peak_x, y=peak_y, mode="markers",
                                 marker={"size": 10, "color": "crimson", "symbol": "diamond"},
                                 name="Peaks", text=peak_text, hoverinfo="text"))

        # Batch all event shapes in one layout call
        shapes = list(fig.layout.shapes or [])
        for i, event in enumerate(events):
            color = _solid_event_color(i)
            x_s = event.start_datetime.isoformat()
            x_e = event.end_datetime.isoformat()
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=x_s, x1=x_s, y0=0, y1=1,
                line=dict(color=color, width=1, dash="longdash"),
            ))
            shapes.append(dict(
                type="line", xref="x", yref="paper",
                x0=x_e, x1=x_e, y0=0, y1=1,
                line=dict(color=color, width=1, dash="dot"),
            ))
            shapes.append(dict(
                type="rect", xref="x", yref="y domain",
                x0=x_s, x1=x_e, y0=0, y1=1,
                fillcolor=_event_color(i, 0.12),
                line=dict(color=color, width=0.5),
                layer="below",
            ))
        fig.update_layout(shapes=shapes)

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

    # Downsampled rainfall bar
    t_rain, rain_ds = _downsample_bar(df["datetime"].values, df["rain"].values)
    fig.add_trace(go.Bar(x=t_rain, y=rain_ds, name="Rainfall",
                         marker_color="steelblue", marker_opacity=0.7), row=1, col=1)
    fig.update_yaxes(autorange="reversed", row=1, col=1)

    obs_df = summary.global_observed.copy()
    obs_df["datetime"] = pd.to_datetime(obs_df["datetime"])
    obs_dt_ns = obs_df["datetime"].values.astype("int64")
    obs_flow = obs_df["stormflow"].values

    fig.add_trace(go.Scatter(x=obs_df["datetime"], y=obs_flow, mode="lines",
                             line={"color": "black", "width": 1.5}, name="Observed"), row=2, col=1)
    fig.add_trace(go.Scatter(x=global_pred["datetime"], y=global_pred["Q_pred"], mode="lines",
                             line={"color": "crimson", "width": 1.5, "dash": "dash"},
                             name="Global predicted"), row=2, col=1)

    # Collect per-event annotation dicts to batch later
    extra_annotations: list[dict] = []

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
            peak_flow = float(obs_flow[_nearest_idx(obs_dt_ns, event.peak_datetime)])
            extra_annotations.append(dict(
                x=event.peak_datetime.isoformat(), y=peak_flow,
                xref="x", yref="y2",
                text="<br>".join(ann_lines), showarrow=True, arrowhead=2,
                arrowcolor=color, font={"size": 9, "color": color},
                bgcolor="white", opacity=0.85,
            ))

    try:
        merged = pd.merge_asof(obs_df.sort_values("datetime"),
                               global_pred.sort_values("datetime"), on="datetime",
                               direction="nearest", tolerance=pd.Timedelta("1h"))
        if not merged.empty:
            r_vals = merged["stormflow"].values - merged["Q_pred"].values
            t_res = merged["datetime"].values
            pos = np.where(r_vals >= 0, r_vals, 0.0)
            neg = np.where(r_vals < 0, r_vals, 0.0)
            # Downsample residual bars (can be 100k+ rows)
            t_pos, pos_ds = _downsample_bar(t_res, pos)
            t_neg, neg_ds = _downsample_bar(t_res, np.abs(neg))
            neg_ds = -neg_ds
            fig.add_trace(go.Bar(x=t_pos, y=pos_ds, name="Residual (+)",
                                 marker_color="rgba(255,100,100,0.6)"), row=3, col=1)
            fig.add_trace(go.Bar(x=t_neg, y=neg_ds, name="Residual (−)",
                                 marker_color="rgba(100,100,255,0.6)"), row=3, col=1)
            fig.add_hline(y=0, line={"color": "grey", "width": 0.8}, row=3, col=1)
    except Exception:
        pass

    # Batch event bands + per-event metric annotations in one layout call
    _add_event_bands(fig, events, rows=[2, 3], show_label=True, label_row=2)
    if extra_annotations:
        all_ann = list(fig.layout.annotations or []) + extra_annotations
        fig.update_layout(annotations=all_ann)

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
    fitted_areas: list[float] | None = None,
    calibrated_areas: list[float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Bar chart comparing observed, fitted, and calibrated effective area per event.

    Parameters
    ----------
    events : list[EventRecord]
        Source of observed Ae and event duration.
    fitted_areas : list[float], optional
        Integrated fitted Ae = ``sum(Q_pred)/sum(rain)`` per event
        (:attr:`SequentialFitSummary.fitted_effective_areas`).
    calibrated_areas : list[float], optional
        Direct calibrated ``A`` parameter per event
        (:meth:`SequentialFitSummary.calibrated_areas`).
        For single models this is the ``A`` value; for ensembles it is the sum
        of all component ``*_A`` parameters.
    title : str, optional
        Auto-generated when omitted.
    """
    if not events:
        return go.Figure().update_layout(title=title or "Effective Area (no events)")

    ids = [e.event_id for e in events]
    x_labels = [f"E{i}" for i in ids]
    obs_ae = np.array([e.effective_area for e in events])
    durations = np.array([e.duration_hours() for e in events])
    mean_ae, std_ae = float(np.mean(obs_ae)), float(np.std(obs_ae))

    has_fitted = fitted_areas is not None
    has_cal = calibrated_areas is not None
    grouped = has_fitted or has_cal

    # Build title
    if title is None:
        parts = [f"obs={mean_ae:.3f}±{std_ae:.3f}"]
        if has_fitted:
            fa = np.array(fitted_areas[: len(events)], dtype=float)
            parts.append(f"fitted={float(np.mean(fa)):.3f}")
        if has_cal:
            ca = np.array(calibrated_areas[: len(events)], dtype=float)
            parts.append(f"cal A={float(np.mean(ca)):.3f}")
        label = "Observed" + (" vs Fitted" if has_fitted else "") + (" vs Calibrated A" if has_cal else "")
        title = f"Effective Area — {label}  ({' | '.join(parts)})"

    fig = go.Figure()
    roll_obs = pd.Series(obs_ae).rolling(window=5, min_periods=1, center=True).mean().values

    # Observed bars — color by duration when shown alone, solid blue when grouped
    if grouped:
        fig.add_trace(go.Bar(
            x=x_labels, y=obs_ae, name="Observed Ae",
            marker_color="steelblue", opacity=0.85,
            text=[f"Obs Ae={v:.3f}<br>Dur={d:.1f}hr" for v, d in zip(obs_ae, durations)],
            hoverinfo="text",
        ))
    else:
        fig.add_trace(go.Bar(
            x=x_labels, y=obs_ae, name="Observed Ae",
            marker={"color": durations, "colorscale": "Blues",
                    "colorbar": {"title": "Duration (hr)", "thickness": 12}, "showscale": True},
            text=[f"Ae={v:.3f}<br>Dur={d:.1f}hr" for v, d in zip(obs_ae, durations)],
            hoverinfo="text",
        ))

    # Fitted Ae (integrated from Q_pred)
    if has_fitted:
        fa = np.array(fitted_areas[: len(events)], dtype=float)
        mean_fit = float(np.mean(fa))
        fig.add_trace(go.Bar(
            x=x_labels, y=fa, name="Fitted Ae  (∫Q_pred/∫rain)",
            marker_color="darkorange", opacity=0.85,
            text=[f"Fit Ae={v:.3f}" for v in fa], hoverinfo="text",
        ))
        fig.add_hline(y=mean_fit,
                      line={"color": "darkorange", "dash": "longdash", "width": 1.2},
                      annotation_text=f"Fit mean={mean_fit:.3f}",
                      annotation_position="bottom right")

    # Calibrated A (direct parameter from optimizer)
    if has_cal:
        ca = np.array(calibrated_areas[: len(events)], dtype=float)
        mean_cal = float(np.mean(ca))
        fig.add_trace(go.Bar(
            x=x_labels, y=ca, name="Calibrated A  (UH param)",
            marker_color="seagreen", opacity=0.85,
            text=[f"Cal A={v:.3f}" for v in ca], hoverinfo="text",
        ))
        fig.add_hline(y=mean_cal,
                      line={"color": "seagreen", "dash": "longdash", "width": 1.2},
                      annotation_text=f"Cal A mean={mean_cal:.3f}",
                      annotation_position="top right")

    # Observed rolling mean + mean line + ±1σ band (always shown)
    fig.add_trace(go.Scatter(
        x=x_labels, y=roll_obs, mode="lines+markers",
        line={"color": "steelblue", "dash": "dash", "width": 2}, marker={"size": 6},
        name="Obs 5-event mean",
    ))
    fig.add_hline(y=mean_ae,
                  line={"color": "steelblue", "dash": "longdash", "width": 1.5},
                  annotation_text=f"Obs mean={mean_ae:.3f}",
                  annotation_position="top left")
    fig.add_hrect(y0=mean_ae - std_ae, y1=mean_ae + std_ae,
                  fillcolor="rgba(70,130,180,0.07)", line_width=0,
                  annotation_text="±1σ obs")

    fig.update_layout(
        title={"text": title, "x": 0.5},
        xaxis_title="Event ID", yaxis_title="Effective Area (Ae)",
        height=450, barmode="group",
        legend={"orientation": "h", "y": -0.18},
    )
    return fig


__all__ = [
    "plot_rainfall_flow_with_events",
    "plot_filter_signals",
    "plot_event_detection",
    "plot_sequential_fit",
    "plot_parameter_evolution",
    "plot_effective_area",
]
