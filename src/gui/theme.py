"""
theme.py
Dark QSS stylesheet, channel colours, and sweep display constants
shared across all GUI panels.
"""

# ── Channel colours ────────────────────────────────────────────────────────────
CHANNEL_COLORS = {
    "TP9":  "#4e9af1",
    "AF7":  "#f1a24e",
    "AF8":  "#4ef18e",
    "TP10": "#f14e7a",
    "AUX":  "#a64ef1",
}

BAND_COLORS = {
    "delta": "#6c8ebf",
    "theta": "#82b366",
    "alpha": "#d6b656",
    "beta":  "#ae4132",
    "gamma": "#9673a6",
}

# ── Sweep display constants ────────────────────────────────────────────────────
SWEEP_WINDOW_S = 5
SWEEP_FS       = 256
SWEEP_SIZE     = SWEEP_WINDOW_S * SWEEP_FS   # 1280 samples
ERASER_WIDTH   = int(0.3 * SWEEP_FS)         # ~77-sample NaN gap (cursor)
CH_OFFSETS     = [300, 150, 0, -150, -300]   # µV offset per channel
CH_YRANGE      = (-500, 500)
CH_NAMES       = ["TP9", "AF7", "AF8", "TP10", "AUX"]

# ── Quality colours ────────────────────────────────────────────────────────────
QUALITY_COLORS = {
    "good":      "#4ef18e",
    "fair":      "#f1d24e",
    "poor":      "#f14e4e",
    "secondary": "#666666",
}

# ── Dark QSS stylesheet ────────────────────────────────────────────────────────
import re


def build_stylesheet(scale: float = 1.0) -> str:
    """Return DARK_QSS with all font-size px values multiplied by `scale`."""
    if scale == 1.0:
        return _DARK_QSS_TEMPLATE
    return re.sub(
        r"font-size:\s*(\d+(?:\.\d+)?)px",
        lambda m: f"font-size: {float(m.group(1)) * scale:g}px",
        _DARK_QSS_TEMPLATE,
    )


_DARK_QSS_TEMPLATE = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QTabWidget::pane {
    border: 1px solid #333355;
    background-color: #16213e;
}

QTabBar::tab {
    background-color: #0f3460;
    color: #aaaacc;
    padding: 8px 20px;
    border: 1px solid #333355;
    border-bottom: none;
    min-width: 110px;
}

QTabBar::tab:selected {
    background-color: #e94560;
    color: #ffffff;
    font-weight: bold;
}

QTabBar::tab:hover:!selected {
    background-color: #1a4a7a;
}

QPushButton {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #333366;
    border-radius: 4px;
    padding: 6px 12px;
    font-size: 12px;
}

QPushButton:hover {
    background-color: #1a4a8a;
    border-color: #5599ff;
}

QPushButton:pressed {
    background-color: #0a2040;
}

QPushButton#btn-connect {
    background-color: #1a6e3c;
    font-size: 14px;
    font-weight: bold;
    padding: 10px;
}

QPushButton#btn-connect:hover {
    background-color: #25a05a;
}

QPushButton#btn-connect[active="true"] {
    background-color: #8b1a1a;
}

QPushButton#btn-connect[active="true"]:hover {
    background-color: #cc2222;
}

QPushButton#btn-record {
    background-color: #6e1a1a;
}

QPushButton#btn-record[recording="true"] {
    background-color: #cc0000;
}

QComboBox {
    background-color: #0f3460;
    color: #e0e0e0;
    border: 1px solid #333366;
    border-radius: 4px;
    padding: 4px 8px;
}

QComboBox::drop-down {
    border: none;
}

QComboBox QAbstractItemView {
    background-color: #0f3460;
    color: #e0e0e0;
    selection-background-color: #1a4a8a;
}

QLabel {
    color: #cccccc;
}

QLabel#title-label {
    font-size: 18px;
    font-weight: bold;
    color: #ffffff;
}

QLabel#status-label {
    font-size: 12px;
    font-weight: bold;
    padding: 4px 8px;
    border-radius: 4px;
}

QLabel#timer-label {
    font-size: 13px;
    color: #aaaaaa;
    font-family: monospace;
}

QCheckBox {
    color: #cccccc;
    spacing: 6px;
}

QCheckBox::indicator {
    width: 14px;
    height: 14px;
    border: 1px solid #555577;
    border-radius: 3px;
    background-color: #0f3460;
}

QCheckBox::indicator:checked {
    background-color: #4e9af1;
    border-color: #4e9af1;
}

QProgressBar {
    border: 2px solid #333355;
    border-radius: 8px;
    background-color: #0a1628;
    text-align: center;
    color: #ffffff;
    font-weight: bold;
    font-size: 14px;
}

QProgressBar::chunk {
    border-radius: 6px;
}

QSplitter::handle {
    background-color: #333355;
    width: 2px;
}

QScrollBar:vertical {
    background: #0f1a2e;
    width: 8px;
}

QScrollBar::handle:vertical {
    background: #333366;
    border-radius: 4px;
    min-height: 20px;
}

QGroupBox {
    border: 1px solid #333366;
    border-radius: 6px;
    margin-top: 10px;
    padding-top: 6px;
    color: #aaaacc;
    font-weight: bold;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
"""

DARK_QSS = build_stylesheet(1.0)
