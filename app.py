"""
app.py
Entry point for the Muse S EEG Dashboard.

Usage
-----
    python app.py

Then open http://localhost:8050 in your browser.
"""

import time
from collections import deque
from datetime import timedelta

import dash
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objects as go
from dash import Input, Output, Patch, State, no_update

from src.data_manager import save_recording
from src.layout import (
    CHANNEL_COLORS, CH_OFFSETS, ERASER_WIDTH, SWEEP_SIZE, SWEEP_WINDOW_S,
    _gauge_figure, build_layout,
)
from src.processing import BANDS, EEGProcessor
from src.stream_reader import CHANNEL_NAMES, StreamReader

# ── Global singletons ─────────────────────────────────────────────────────────
_reader    = StreamReader()
_processor = EEGProcessor(notch_freq=50.0)

# ── Sweep display state ───────────────────────────────────────────────────────
N_CH           = 5
_y_sweep       = np.full((SWEEP_SIZE, N_CH), np.nan)   # circular display buffer
_write_idx     = 0                                       # next write position
_last_seen     = 0                                       # sample_count consumed
_filter_states = _processor.make_filter_states(N_CH)    # causal filter memory

# Band power rolling history: {band: deque of (timestamp, avg_power)}
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


# ── Figure helpers (slow path only) ──────────────────────────────────────────

def _dark_layout(yaxis_title: str = "") -> dict:
    return {
        "plot_bgcolor":  "#111",
        "paper_bgcolor": "#111",
        "font":          {"color": "#ccc"},
        "yaxis":         {"title": yaxis_title, "gridcolor": "#2a2a2a", "color": "#777"},
    }


def _bands_bar_fig(powers: dict) -> go.Figure:
    band_colors = {
        "delta": "#6c8ebf", "theta": "#82b366", "alpha": "#d6b656",
        "beta": "#ae4132", "gamma": "#9673a6",
    }
    fig = go.Figure()
    for ch_i, ch in enumerate(CHANNEL_NAMES[:4]):
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
        uirevision="bands-bar",
    )
    return fig


def _bands_history_fig() -> go.Figure:
    band_colors = {
        "delta": "#6c8ebf", "theta": "#82b366", "alpha": "#d6b656",
        "beta": "#ae4132", "gamma": "#9673a6",
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


def _quality_to_color(label: str) -> str:
    return {"good": "success", "fair": "warning", "poor": "danger"}.get(label, "secondary")


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


# ── Callback: stream toggle ───────────────────────────────────────────────────

@app.callback(
    Output("btn-stream-toggle",   "children"),
    Output("btn-stream-toggle",   "color"),
    Output("connection-badge",    "children"),
    Output("connection-badge",    "color"),
    Output("interval-eeg",        "disabled"),
    Output("interval-slow",       "disabled"),
    Output("interval-timer",      "disabled"),
    Output("store-stream-active", "data"),
    Input("btn-stream-toggle",    "n_clicks"),
    State("store-stream-active",  "data"),
    State("device-dropdown",      "value"),
    prevent_initial_call=True,
)
def toggle_stream(n_clicks, is_active, selected_address):
    global _y_sweep, _write_idx, _last_seen, _filter_states
    if not is_active:
        ok = _reader.start(address=selected_address or None)
        if ok:
            # Reset sweep state for fresh connection
            _y_sweep[:] = np.nan
            _write_idx  = 0
            _last_seen  = _reader.sample_count
            _filter_states = _processor.make_filter_states(N_CH)
            label = f"⏹ Disconnect ({_reader.device_name})"
            return label, "danger", "● LIVE", "success", False, False, False, True
        else:
            err = _reader.last_error or "BLE connection failed"
            return "▶ Connect", "success", f"● FAILED: {err}", "danger", True, True, True, False
    else:
        _reader.stop()
        return "▶ Connect", "success", "● OFFLINE", "secondary", True, True, True, False


# ── Callback: notch filter change ─────────────────────────────────────────────

@app.callback(
    Output("store-notch", "data"),
    Input("notch-selector", "value"),
    prevent_initial_call=True,
)
def update_notch(freq):
    from src.processing import _notch_sos
    _processor._notch_sos = _notch_sos(freq, _processor.fs)
    # Reset causal filter states so transient from old filter clears
    global _filter_states
    _filter_states = _processor.make_filter_states(N_CH)
    return freq


# ── Callback: recording toggle ─────────────────────────────────────────────────

@app.callback(
    Output("btn-record-toggle",     "children"),
    Output("btn-record-toggle",     "color"),
    Output("btn-save",              "disabled"),
    Output("store-recording-active","data"),
    Output("store-record-start",    "data"),
    Input("btn-record-toggle",      "n_clicks"),
    State("store-recording-active", "data"),
    State("store-stream-active",    "data"),
    prevent_initial_call=True,
)
def toggle_recording(n_clicks, is_recording, stream_active):
    if not stream_active:
        return no_update, no_update, no_update, no_update, no_update
    if not is_recording:
        return "⏹ Stop Recording", "danger", True, True, time.time()
    return "⏺ Start Recording", "danger", False, False, no_update


# ── Callback: save data ───────────────────────────────────────────────────────

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
    save_recording(ts, data, sample_rate=_reader.sample_rate, metric_history=metric_h)
    return f"✓ Saved ({data.shape[0]} samples)"


# ── Callback: recording timer ─────────────────────────────────────────────────

@app.callback(
    Output("recording-timer", "children"),
    Input("interval-timer",   "n_intervals"),
    State("store-recording-active", "data"),
    State("store-record-start",     "data"),
)
def update_timer(n, is_recording, start_time):
    if not is_recording or start_time is None:
        return "00:00:00"
    return str(timedelta(seconds=int(time.time() - start_time)))


# ── Callback: fast EEG sweep (10 Hz) ─────────────────────────────────────────

@app.callback(
    Output("graph-raw-eeg", "figure"),
    Output("quality-tp9",   "color"),
    Output("quality-af7",   "color"),
    Output("quality-af8",   "color"),
    Output("quality-tp10",  "color"),
    Input("interval-eeg",   "n_intervals"),
    State("toggle-tp9",     "value"),
    State("toggle-af7",     "value"),
    State("toggle-af8",     "value"),
    State("toggle-tp10",    "value"),
    State("toggle-aux",     "value"),
    prevent_initial_call=True,
)
def update_eeg_sweep(n, tp9, af7, af8, tp10, aux):
    global _y_sweep, _write_idx, _last_seen, _filter_states

    # 1. Pull only new samples since last tick
    new_raw, _ = _reader.get_new_samples(_last_seen)
    n_new = new_raw.shape[0]
    _last_seen += n_new

    if n_new > 0:
        # 2. Causal filter — fast, only the new samples
        filtered, _filter_states = _processor.filter_new(new_raw, _filter_states)

        # 3. Write into circular sweep buffer (handle wrap-around)
        end = _write_idx + n_new
        if end <= SWEEP_SIZE:
            _y_sweep[_write_idx:end] = filtered
        else:
            first = SWEEP_SIZE - _write_idx
            _y_sweep[_write_idx:] = filtered[:first]
            _y_sweep[:end - SWEEP_SIZE] = filtered[first:]
        _write_idx = end % SWEEP_SIZE

        # 4. Stamp eraser gap (NaN) at write head — the moving "cursor"
        eraser_end = min(_write_idx + ERASER_WIDTH, SWEEP_SIZE)
        _y_sweep[_write_idx:eraser_end] = np.nan
        wrap = (_write_idx + ERASER_WIDTH) - SWEEP_SIZE
        if wrap > 0:
            _y_sweep[:wrap] = np.nan

    # 5. Signal quality from non-NaN region
    valid = ~np.isnan(_y_sweep[:, :4])
    if valid.any():
        safe = np.where(valid, _y_sweep[:, :4], 0.0)
        quality = _processor.signal_quality(safe)
    else:
        quality = ["poor"] * 4
    badge_colors = [_quality_to_color(q) for q in quality]

    # 6. Build Patch — update only the y-arrays of existing traces
    show = [tp9, af7, af8, tp10, aux]
    patched = Patch()
    for i in range(N_CH):
        if show[i]:
            y = _y_sweep[:, i].copy()
            mask = ~np.isnan(y)
            y[mask] += CH_OFFSETS[i]
            patched["data"][i]["y"] = y.tolist()
        else:
            patched["data"][i]["y"] = [float("nan")] * SWEEP_SIZE

    return patched, *badge_colors


# ── Callback: slow band powers + mental metrics (0.5 Hz) ─────────────────────

@app.callback(
    Output("graph-bands-bar",        "figure"),
    Output("graph-bands-history",    "figure"),
    Output("gauge-workload",         "figure"),
    Output("gauge-fatigue",          "figure"),
    Output("graph-workload-history", "figure"),
    Output("graph-fatigue-history",  "figure"),
    Input("interval-slow",           "n_intervals"),
    State("main-tabs",               "active_tab"),
    prevent_initial_call=True,
)
def update_slow(n, active_tab):
    # Get latest 4 s epoch
    epoch_raw, _ = _reader.get_window(seconds=4)
    if epoch_raw.shape[0] < int(_reader.sample_rate):
        return no_update, no_update, no_update, no_update, no_update, no_update

    # Zero-phase filter on epoch (fine at 0.5 Hz)
    epoch = _processor.filter_signal(epoch_raw)

    # Single Welch PSD — shared by both the bar chart and mental metrics
    powers = _processor.band_powers(epoch)

    # Update band history
    now = time.time()
    for band in BANDS:
        _band_history[band].append((now, float(np.mean(powers[band][:4]))))

    # Frontal mean powers (AF7=idx1, AF8=idx2)
    frontal_powers = {
        b: float(np.mean(powers[b][[1, 2]]))
        for b in ("theta", "alpha", "beta")
    }
    metrics = _processor.compute_metrics_from_powers(frontal_powers, now)

    # Only rebuild figures for the currently visible tab
    if active_tab == "tab-bands":
        bars_fig = _bands_bar_fig(powers)
        hist_fig = _bands_history_fig()
    else:
        bars_fig = hist_fig = no_update

    if active_tab == "tab-mental":
        gauge_wl = _gauge_figure(metrics["workload"], "Workload", "#f1a24e")
        gauge_fa = _gauge_figure(metrics["fatigue"],  "Fatigue",  "#f14e7a")
        hist_wl  = _metric_history_fig("workload", "rgb(241,162,78)", "Mental Workload")
        hist_fa  = _metric_history_fig("fatigue",  "rgb(241,78,122)", "Mental Fatigue")
    else:
        gauge_wl = gauge_fa = hist_wl = hist_fa = no_update

    return bars_fig, hist_fig, gauge_wl, gauge_fa, hist_wl, hist_fa


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  🧠  Muse S EEG Dashboard")
    print("=" * 60)
    print("  Open your browser at:  http://localhost:8050")
    print("  Press Ctrl+C to stop.\n")
    app.run(debug=False, host="localhost", port=8050)
