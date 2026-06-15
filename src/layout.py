"""
layout.py
Dash UI layout for the Muse S EEG Dashboard.
"""

import dash_bootstrap_components as dbc
from dash import dcc, html

CHANNEL_COLORS = {
    "TP9":  "#4e9af1",
    "AF7":  "#f1a24e",
    "AF8":  "#4ef18e",
    "TP10": "#f14e7a",
    "AUX":  "#a64ef1",
}

# ── Status indicator helper ─────────────────────────────────────────────────────

def _quality_badge(channel: str) -> dbc.Badge:
    return dbc.Badge(
        channel,
        id=f"quality-{channel.lower()}",
        color="secondary",
        className="me-1 fs-6",
    )


# ── Header ───────────────────────────────────────────────────────────────────────

def _header() -> dbc.Navbar:
    return dbc.Navbar(
        dbc.Container([
            html.Div([
                html.Img(
                    src="https://upload.wikimedia.org/wikipedia/commons/thumb/8/8e/Muse_EEG_headband.png/100px-Muse_EEG_headband.png",
                    height="36px",
                    className="me-2",
                    style={"display": "none"},   # hidden if img unavailable
                ),
                dbc.NavbarBrand("🧠 Muse S EEG Dashboard", className="fw-bold fs-4"),
            ], className="d-flex align-items-center"),
            html.Div([
                dbc.Badge("● DEMO", id="connection-badge", color="warning",
                          className="me-2 fs-6 px-3 py-2"),
                html.Small(id="recording-timer", children="00:00:00",
                           className="text-white-50 me-3 fs-6"),
            ], className="d-flex align-items-center"),
        ], fluid=True),
        color="dark",
        dark=True,
        className="mb-0 py-2",
    )


# ── Control sidebar ──────────────────────────────────────────────────────────────

def _sidebar() -> dbc.Col:
    return dbc.Col([
        dbc.Card([
            dbc.CardHeader("⚙️ Controls", className="fw-bold"),
            dbc.CardBody([
                # Stream toggle
                dbc.Button(
                    "▶ Connect to Muse",
                    id="btn-stream-toggle",
                    color="success",
                    className="w-100 mb-3",
                    size="lg",
                ),
                html.Hr(),
                # Recording
                dbc.Button(
                    "⏺ Start Recording",
                    id="btn-record-toggle",
                    color="danger",
                    outline=True,
                    className="w-100 mb-2",
                    size="lg",
                ),
                dbc.Button(
                    "💾 Save Data",
                    id="btn-save",
                    color="primary",
                    outline=True,
                    className="w-100 mb-3",
                    size="lg",
                    disabled=True,
                ),
                html.Div(id="save-status", className="text-success small mb-3"),
                html.Hr(),
                # Settings
                html.P("Display window", className="mb-1 text-muted small fw-bold"),
                dcc.Slider(
                    id="window-slider",
                    min=2, max=30, step=1, value=5,
                    marks={2: "2s", 10: "10s", 20: "20s", 30: "30s"},
                    tooltip={"placement": "bottom", "always_visible": True},
                ),
                html.Hr(),
                html.P("Notch filter", className="mb-1 text-muted small fw-bold"),
                dbc.RadioItems(
                    id="notch-selector",
                    options=[
                        {"label": "50 Hz (Europe)", "value": 50},
                        {"label": "60 Hz (US/CA)",  "value": 60},
                    ],
                    value=50,
                    className="mb-3",
                ),
                html.Hr(),
                # Channel quality
                html.P("Signal quality", className="mb-2 text-muted small fw-bold"),
                html.Div([
                    _quality_badge("TP9"),
                    _quality_badge("AF7"),
                    _quality_badge("AF8"),
                    _quality_badge("TP10"),
                ], className="d-flex flex-wrap gap-1"),
            ]),
        ], className="shadow-sm"),

        # Hidden stores
        dcc.Store(id="store-stream-active",    data=False),
        dcc.Store(id="store-recording-active", data=False),
        dcc.Store(id="store-record-start",     data=None),
        dcc.Store(id="store-notch",            data=50),

        # Update interval
        dcc.Interval(id="interval-update", interval=200, n_intervals=0, disabled=True),
        dcc.Interval(id="interval-timer",   interval=1000, n_intervals=0, disabled=True),

    ], width=2, className="pe-1")


# ── Raw EEG tab ──────────────────────────────────────────────────────────────────

def _raw_eeg_tab() -> dbc.Tab:
    ch_toggles = [
        dbc.Switch(
            id=f"toggle-{ch.lower()}",
            label=ch,
            value=True,
            className="me-3",
            style={"color": CHANNEL_COLORS[ch]},
        )
        for ch in ["TP9", "AF7", "AF8", "TP10"]
    ]
    return dbc.Tab(
        label="📈 Raw EEG",
        tab_id="tab-raw",
        children=[
            dbc.Card([
                dbc.CardBody([
                    html.Div(
                        ch_toggles + [
                            dbc.Switch(id="toggle-aux", label="AUX", value=False, className="me-3"),
                        ],
                        className="d-flex flex-wrap mb-3",
                    ),
                    dcc.Graph(
                        id="graph-raw-eeg",
                        config={"displayModeBar": False},
                        style={"height": "480px"},
                        figure=_empty_figure("Waiting for stream…", "µV"),
                    ),
                ]),
            ], className="shadow-sm mt-2"),
        ],
    )


# ── EEG Bands tab ────────────────────────────────────────────────────────────────

def _bands_tab() -> dbc.Tab:
    return dbc.Tab(
        label="🌊 EEG Bands",
        tab_id="tab-bands",
        children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(
                                id="graph-bands-bar",
                                config={"displayModeBar": False},
                                style={"height": "400px"},
                                figure=_empty_figure("Waiting for stream…"),
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="graph-bands-radar",
                                config={"displayModeBar": False},
                                style={"height": "400px"},
                                figure=_empty_figure(""),
                            ),
                            width=6,
                        ),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(
                                id="graph-bands-history",
                                config={"displayModeBar": False},
                                style={"height": "300px"},
                                figure=_empty_figure("Band power over time"),
                            ),
                            width=12,
                        ),
                    ], className="mt-2"),
                ]),
            ], className="shadow-sm mt-2"),
        ],
    )


# ── Mental State tab ─────────────────────────────────────────────────────────────

def _mental_state_tab() -> dbc.Tab:
    return dbc.Tab(
        label="🧠 Mental State",
        tab_id="tab-mental",
        children=[
            dbc.Card([
                dbc.CardBody([
                    dbc.Row([
                        # Workload gauge + history
                        dbc.Col([
                            html.H5("Mental Workload", className="text-center fw-bold"),
                            html.P(
                                "Frontal θ+α / α+β ratio — rises with cognitive effort",
                                className="text-muted small text-center mb-2",
                            ),
                            dcc.Graph(
                                id="gauge-workload",
                                config={"displayModeBar": False},
                                style={"height": "220px"},
                                figure=_gauge_figure(0, "Workload", "#f1a24e"),
                            ),
                        ], width=6),
                        # Fatigue gauge + history
                        dbc.Col([
                            html.H5("Mental Fatigue", className="text-center fw-bold"),
                            html.P(
                                "Frontal θ / α ratio — rises with sustained mental effort",
                                className="text-muted small text-center mb-2",
                            ),
                            dcc.Graph(
                                id="gauge-fatigue",
                                config={"displayModeBar": False},
                                style={"height": "220px"},
                                figure=_gauge_figure(0, "Fatigue", "#f14e7a"),
                            ),
                        ], width=6),
                    ]),
                    dbc.Row([
                        dbc.Col(
                            dcc.Graph(
                                id="graph-workload-history",
                                config={"displayModeBar": False},
                                style={"height": "260px"},
                                figure=_empty_figure("Mental Workload over time", "%"),
                            ),
                            width=6,
                        ),
                        dbc.Col(
                            dcc.Graph(
                                id="graph-fatigue-history",
                                config={"displayModeBar": False},
                                style={"height": "260px"},
                                figure=_empty_figure("Mental Fatigue over time", "%"),
                            ),
                            width=6,
                        ),
                    ], className="mt-2"),
                ]),
            ], className="shadow-sm mt-2"),
        ],
    )


# ── Main layout ───────────────────────────────────────────────────────────────────

def build_layout() -> html.Div:
    return html.Div([
        _header(),
        dbc.Container([
            dbc.Row([
                _sidebar(),
                dbc.Col([
                    dbc.Tabs(
                        id="main-tabs",
                        active_tab="tab-raw",
                        children=[
                            _raw_eeg_tab(),
                            _bands_tab(),
                            _mental_state_tab(),
                        ],
                        className="mt-2",
                    ),
                ], width=10, className="ps-1"),
            ]),
        ], fluid=True, className="mt-2"),
    ])


# ── Figure helpers ────────────────────────────────────────────────────────────────

def _empty_figure(title: str = "", yaxis_label: str = "") -> dict:
    return {
        "data": [],
        "layout": {
            "title":    {"text": title, "font": {"color": "#aaa"}},
            "xaxis":    {"color": "#555", "gridcolor": "#2a2a2a", "showgrid": True},
            "yaxis":    {"title": yaxis_label, "color": "#555", "gridcolor": "#2a2a2a"},
            "plot_bgcolor":  "#111",
            "paper_bgcolor": "#111",
            "font":          {"color": "#ccc"},
            "margin":        {"l": 50, "r": 20, "t": 40, "b": 40},
            "annotations": [{
                "text":      title or "No data yet",
                "xref":      "paper",
                "yref":      "paper",
                "x":         0.5,
                "y":         0.5,
                "showarrow": False,
                "font":      {"color": "#555", "size": 16},
            }] if not title.startswith("Band") else [],
        },
    }


def _gauge_figure(value: float, title: str, color: str) -> dict:
    import plotly.graph_objects as go
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        delta={"reference": 50},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": "#aaa"},
            "bar":  {"color": color},
            "bgcolor": "#1a1a1a",
            "steps": [
                {"range": [0,  33], "color": "#1a2a1a"},
                {"range": [33, 66], "color": "#2a2a1a"},
                {"range": [66, 100], "color": "#2a1a1a"},
            ],
            "threshold": {
                "line": {"color": "white", "width": 2},
                "thickness": 0.8,
                "value": 75,
            },
        },
        number={"suffix": "%", "font": {"color": "#eee", "size": 36}},
    ))
    fig.update_layout(
        height=220,
        paper_bgcolor="#111",
        font={"color": "#ccc"},
        margin={"l": 20, "r": 20, "t": 10, "b": 10},
    )
    return fig
