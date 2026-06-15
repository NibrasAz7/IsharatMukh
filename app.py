"""
app.py
Entry point for the Muse S EEG Dashboard.

Usage
-----
    python app.py

Then open http://localhost:8050 in your browser.
"""

import time
from datetime import datetime, timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, State, callback_context, no_update

from src.data_manager import save_recording
from src.layout import CHANNEL_COLORS, _gauge_figure, build_layout
from src.processing import BANDS, EEGProcessor
from src.stream_reader import CHANNEL_NAMES, StreamReader

# ── Global state ─────────────────────────────────────────────────────────────────
# These live for the lifetime of the server process.
_reader    = StreamReader()
_processor = EEGProcessor(notch_freq=50.0)

# Band power rolling history: {band: [(timestamp, avg_power), ...]}
from collections import deque
_band_history: dict[str, deque] = {b: deque(maxlen=3600) for b in BANDS}

# ── App init ──────────────────────────────────────────────────────────────────────
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.CYBORG],
    title="Muse S EEG Dashboard",
    update_title=None,
)
app.layout = build_layout()
server = app.server   # expose for potential WSGI deployment


# ── Helper: build raw EEG figure ─────────────────────────────────────────────────

def _raw_fig(data: np.ndarray, timestamps: np.ndarray,
             show_channels: list[bool], window_s: float) -> go.Figure:
    ch_names = ["TP9", "AF7", "AF8", "TP10", "AUX"]
    fig = go.Figure()

    if data.shape[0] == 0:
        fig.update_layout(**_dark_layout("µV"))
        return fig

    t_rel = timestamps - timestamps[0]   # relative seconds

    for i, ch in enumerate(ch_names):
        if i < len(show_channels) and show_channels[i]:
            fig.add_trace(go.Scattergl(
                x=t_rel,
                y=data[:, i],
                name=ch,
                line={"color": CHANNEL_COLORS.get(ch, "#fff"), "width": 1},
                mode="lines",
            ))

    fig.update_layout(
        **_dark_layout("µV"),
        xaxis_title="Time (s)",
        xaxis={"range": [max(0, t_rel[-1] - window_s), t_rel[-1]],
               "gridcolor": "#2a2a2a", "color": "#777"},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.01},
        margin={"l": 55, "r": 10, "t": 10, "b": 40},
        uirevision="raw-eeg",
    )
    return fig


# ── Helper: band power figures ────────────────────────────────────────────────────

def _bands_bar_fig(powers: dict[str, np.ndarray]) -> go.Figure:
    band_colors = {
        "delta": "#6c8ebf",
        "theta": "#82b366",
        "alpha": "#d6b656",
        "beta":  "#ae4132",
        "gamma": "#9673a6",
    }
    fig = go.Figure()
    ch_names = CHANNEL_NAMES[:4]
    for ch_i, ch in enumerate(ch_names):
        fig.add_trace(go.Bar(
            name=ch,
            x=list(BANDS.keys()),
            y=[float(powers[b][ch_i]) if b in powers and len(powers[b]) > ch_i else 0
               for b in BANDS],
            marker_color=CHANNEL_COLORS.get(ch, "#aaa"),
        ))
    fig.update_layout(
        **_dark_layout("µV²"),
        barmode="group",
        title={"text": "Current Band Power", "font": {"color": "#ccc"}},
        legend={"orientation": "h"},
        margin={"l": 55, "r": 10, "t": 40, "b": 40},
        xaxis={"color": "#777"},
    )
    return fig


def _bands_radar_fig(powers: dict[str, np.ndarray]) -> go.Figure:
    band_names = list(BANDS.keys())
    fig = go.Figure()
    for ch_i, ch in enumerate(CHANNEL_NAMES[:4]):
        vals = [float(powers[b][ch_i]) if b in powers and len(powers[b]) > ch_i else 0
                for b in band_names]
        vals_norm = _minmax_norm(vals)
        fig.add_trace(go.Scatterpolar(
            r=vals_norm + [vals_norm[0]],
            theta=band_names + [band_names[0]],
            name=ch,
            line={"color": CHANNEL_COLORS.get(ch, "#aaa")},
            fill="toself",
            fillcolor=CHANNEL_COLORS.get(ch, "#aaa"),
            opacity=0.25,
        ))
    fig.update_layout(
        polar={
            "bgcolor": "#1a1a1a",
            "radialaxis": {"visible": True, "color": "#555", "gridcolor": "#333"},
            "angularaxis": {"color": "#777"},
        },
        paper_bgcolor="#111",
        font={"color": "#ccc"},
        margin={"l": 20, "r": 20, "t": 30, "b": 20},
        title={"text": "Band Profile", "font": {"color": "#ccc"}},
        legend={"orientation": "h"},
        uirevision="radar",
    )
    return fig


def _bands_history_fig() -> go.Figure:
    band_colors = {
        "delta": "#6c8ebf",
        "theta": "#82b366",
        "alpha": "#d6b656",
        "beta":  "#ae4132",
        "gamma": "#9673a6",
    }
    fig = go.Figure()
    for band, hist in _band_history.items():
        if not hist:
            continue
        ts  = [h[0] for h in hist]
        val = [h[1] for h in hist]
        t_rel = [t - ts[0] for t in ts]
        fig.add_trace(go.Scattergl(
            x=t_rel, y=val, name=band,
            line={"color": band_colors.get(band, "#aaa"), "width": 1.5},
            mode="lines",
        ))
    fig.update_layout(
        **_dark_layout("µV²"),
        title={"text": "Band Power History", "font": {"color": "#ccc"}},
        xaxis_title="Time (s)",
        legend={"orientation": "h"},
        margin={"l": 55, "r": 10, "t": 40, "b": 40},
        uirevision="band-hist",
    )
    return fig


# ── Helper: metric history figures ───────────────────────────────────────────────

def _metric_history_fig(metric: str, color: str, title: str) -> go.Figure:
    ts, vals = _processor.get_metric_history(metric)
    fig = go.Figure()
    if len(ts) > 0:
        t_rel = ts - ts[0]
        fig.add_trace(go.Scattergl(
            x=t_rel, y=vals,
            line={"color": color, "width": 2},
            mode="lines",
            fill="tozeroy",
            fillcolor=color.replace(")", ",0.15)").replace("rgb", "rgba"),
            name=title,
        ))
        # Add threshold line at 75
        fig.add_hline(y=75, line_dash="dash", line_color="#ff6666",
                      annotation_text="High", annotation_position="right")
    fig.update_layout(
        **_dark_layout("%"),
        title={"text": title, "font": {"color": "#ccc"}},
        xaxis_title="Time (s)",
        yaxis={"range": [0, 100], "gridcolor": "#2a2a2a", "color": "#777"},
        margin={"l": 55, "r": 60, "t": 40, "b": 40},
        uirevision=f"metric-{metric}",
    )
    return fig


# ── Callback: scan for BLE devices ──────────────────────────────────────────────

@app.callback(
    Output("scan-status",        "children"),
    Output("btn-scan",           "disabled"),
    Output("interval-scan-poll", "disabled"),
    Input("btn-scan",            "n_clicks"),
    State("store-stream-active", "data"),
    prevent_initial_call=True,
)
def start_scan(n_clicks, stream_active):
    if stream_active:
        return "Disconnect first.", False, True
    _reader.scan_devices()   # non-blocking background thread
    return "🔍 Scanning (10s)…", True, False


@app.callback(
    Output("device-dropdown",    "options"),
    Output("device-dropdown",    "value"),
    Output("scan-status",        "children", allow_duplicate=True),
    Output("btn-scan",           "disabled", allow_duplicate=True),
    Output("interval-scan-poll", "disabled", allow_duplicate=True),
    Input("interval-scan-poll",  "n_intervals"),
    prevent_initial_call=True,
)
def poll_scan_results(n):
    if _reader.is_scanning:
        return no_update, no_update, no_update, no_update, no_update
    devices = _reader.discovered_devices
    if not devices:
        return [], None, "⚠ No Muse found. Try again.", False, True
    options = [
        {"label": f"{d['name']}  ({d['address']})", "value": d["address"]}
        for d in devices
    ]
    return options, options[0]["value"], f"✓ {len(devices)} device(s) found", False, True


# ── Callback: stream toggle ───────────────────────────────────────────────────────

@app.callback(
    Output("btn-stream-toggle",  "children"),
    Output("btn-stream-toggle",  "color"),
    Output("connection-badge",   "children"),
    Output("connection-badge",   "color"),
    Output("interval-update",    "disabled"),
    Output("interval-timer",     "disabled"),
    Output("store-stream-active", "data"),
    Input("btn-stream-toggle",   "n_clicks"),
    State("store-stream-active", "data"),
    State("device-dropdown",     "value"),
    prevent_initial_call=True,
)
def toggle_stream(n_clicks, is_active, selected_address):
    if not is_active:
        _reader.start(address=selected_address or None)
        if _reader.is_demo:
            badge_text, badge_color = "● DEMO", "warning"
            label = "⏹ Disconnect (demo)"
        else:
            badge_text, badge_color = "● LIVE", "success"
            label = f"⏹ Disconnect ({_reader.device_name})"
        return label, "danger", badge_text, badge_color, False, False, True
    else:
        _reader.stop()
        return "▶ Connect", "success", "● OFFLINE", "secondary", True, True, False


# ── Callback: notch filter change ────────────────────────────────────────────────

@app.callback(
    Output("store-notch", "data"),
    Input("notch-selector", "value"),
    prevent_initial_call=True,
)
def update_notch(freq):
    from scipy.signal import iirnotch, tf2sos
    from src.processing import _notch_sos
    _processor._notch_sos = _notch_sos(freq, _processor.fs)
    return freq


# ── Callback: recording toggle ────────────────────────────────────────────────────

@app.callback(
    Output("btn-record-toggle",    "children"),
    Output("btn-record-toggle",    "color"),
    Output("btn-save",             "disabled"),
    Output("store-recording-active", "data"),
    Output("store-record-start",   "data"),
    Input("btn-record-toggle",     "n_clicks"),
    State("store-recording-active", "data"),
    State("store-stream-active",   "data"),
    prevent_initial_call=True,
)
def toggle_recording(n_clicks, is_recording, stream_active):
    if not stream_active:
        return no_update, no_update, no_update, no_update, no_update
    if not is_recording:
        return "⏹ Stop Recording", "danger", True, True, time.time()
    else:
        return "⏺ Start Recording", "danger", False, False, no_update


# ── Callback: save data ───────────────────────────────────────────────────────────

@app.callback(
    Output("save-status", "children"),
    Input("btn-save",     "n_clicks"),
    prevent_initial_call=True,
)
def save_data(n_clicks):
    data, ts = _reader.get_all()
    if data.shape[0] == 0:
        return "⚠ No data to save."
    metric_h = {
        "workload": list(_processor.metric_history["workload"]),
        "fatigue":  list(_processor.metric_history["fatigue"]),
    }
    paths = save_recording(ts, data, sample_rate=_reader.sample_rate,
                           metric_history=metric_h)
    return f"✓ Saved ({data.shape[0]} samples)"


# ── Callback: recording timer ─────────────────────────────────────────────────────

@app.callback(
    Output("recording-timer", "children"),
    Input("interval-timer",   "n_intervals"),
    State("store-recording-active", "data"),
    State("store-record-start",     "data"),
)
def update_timer(n, is_recording, start_time):
    if not is_recording or start_time is None:
        return "00:00:00"
    elapsed = int(time.time() - start_time)
    return str(timedelta(seconds=elapsed))


# ── Main update callback ──────────────────────────────────────────────────────────

@app.callback(
    # Raw EEG
    Output("graph-raw-eeg",        "figure"),
    # Signal quality badges
    Output("quality-tp9",          "color"),
    Output("quality-af7",          "color"),
    Output("quality-af8",          "color"),
    Output("quality-tp10",         "color"),
    # Band power
    Output("graph-bands-bar",      "figure"),
    Output("graph-bands-radar",    "figure"),
    Output("graph-bands-history",  "figure"),
    # Mental state
    Output("gauge-workload",       "figure"),
    Output("gauge-fatigue",        "figure"),
    Output("graph-workload-history","figure"),
    Output("graph-fatigue-history", "figure"),
    # Inputs
    Input("interval-update",       "n_intervals"),
    State("window-slider",         "value"),
    State("toggle-tp9",            "value"),
    State("toggle-af7",            "value"),
    State("toggle-af8",            "value"),
    State("toggle-tp10",           "value"),
    State("toggle-aux",            "value"),
    State("main-tabs",             "active_tab"),
    prevent_initial_call=True,
)
def update_all(n_intervals, window_s, tp9, af7, af8, tp10, aux, active_tab):
    show = [tp9, af7, af8, tp10, aux]

    # Get raw window
    raw, ts = _reader.get_window(seconds=max(window_s, 4))
    if raw.shape[0] == 0:
        empty = _empty_raw_fig()
        return (empty, "secondary", "secondary", "secondary", "secondary",
                *[no_update] * 8)

    # Filter
    filtered = _processor.filter_signal(raw)

    # Signal quality
    quality = _processor.signal_quality(filtered[:, :4])
    badge_colors = [_quality_to_color(q) for q in quality]

    # Raw EEG figure (always update)
    raw_fig = _raw_fig(filtered, ts, show, window_s)

    # Band powers (compute from latest 4 s epoch)
    epoch_samples = int(_reader.sample_rate * 4)
    epoch = filtered[-epoch_samples:, :] if filtered.shape[0] >= epoch_samples else filtered
    powers = _processor.band_powers(epoch)

    # Update band history
    now = time.time()
    for band in BANDS:
        avg_power = float(np.mean(powers[band][:4]))
        _band_history[band].append((now, avg_power))

    # Mental metrics
    metrics = _processor.compute_metrics(epoch, now)

    # Figures
    bands_bar_fig    = _bands_bar_fig(powers) if active_tab == "tab-bands"   else no_update
    bands_radar_fig  = _bands_radar_fig(powers) if active_tab == "tab-bands" else no_update
    bands_hist_fig   = _bands_history_fig() if active_tab == "tab-bands"     else no_update

    workload_val = metrics["workload"]
    fatigue_val  = metrics["fatigue"]

    if active_tab == "tab-mental":
        gauge_wl  = _gauge_figure(workload_val, "Workload", "#f1a24e")
        gauge_fa  = _gauge_figure(fatigue_val,  "Fatigue",  "#f14e7a")
        hist_wl   = _metric_history_fig("workload", "rgb(241,162,78)",  "Mental Workload")
        hist_fa   = _metric_history_fig("fatigue",  "rgb(241,78,122)",  "Mental Fatigue")
    else:
        gauge_wl = gauge_fa = hist_wl = hist_fa = no_update

    return (
        raw_fig,
        *badge_colors,
        bands_bar_fig, bands_radar_fig, bands_hist_fig,
        gauge_wl, gauge_fa, hist_wl, hist_fa,
    )


# ── Utilities ─────────────────────────────────────────────────────────────────────

def _dark_layout(yaxis_title: str = "") -> dict:
    return {
        "plot_bgcolor":  "#111",
        "paper_bgcolor": "#111",
        "font":          {"color": "#ccc"},
        "yaxis":         {"title": yaxis_title, "gridcolor": "#2a2a2a", "color": "#777"},
    }


def _empty_raw_fig() -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        **_dark_layout("µV"),
        annotations=[{
            "text": "Waiting for EEG data…",
            "xref": "paper", "yref": "paper",
            "x": 0.5, "y": 0.5,
            "showarrow": False,
            "font": {"color": "#555", "size": 18},
        }],
        margin={"l": 55, "r": 10, "t": 10, "b": 40},
    )
    return fig


def _quality_to_color(label: str) -> str:
    return {"good": "success", "fair": "warning", "poor": "danger"}.get(label, "secondary")


def _minmax_norm(vals: list) -> list:
    mn, mx = min(vals), max(vals)
    if mx == mn:
        return [0.5] * len(vals)
    return [(v - mn) / (mx - mn) for v in vals]


# ── Entry point ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠  Muse S EEG Dashboard")
    print("=" * 60)
    print("  Open your browser at:  http://localhost:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="localhost", port=8050)
