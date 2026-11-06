"""
mental_panel.py
Mental State tab — 5 EEG-based metrics with gauges, stacked history plots,
dynamic transparency, and linked zoom/pan.

Metrics (peer-reviewed formulas):
  Workload   : (θ+α)/(α+β)        — Oken et al. 2006
  Fatigue    : θ/α                 — Oken et al. 2006
  Stress     : β/α                 — Alarcao & Fonseca 2017
  Attention  : β/(θ+α)            — Pope et al. 1995; Sterman 1996
  FAA        : ln(α_AF8)−ln(α_AF7) — Davidson 1990; Harmon-Jones & Allen 1998
               50=neutral, >50=approach, <50=withdrawal
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QScrollArea,
    QSplitter, QVBoxLayout, QWidget,
)

# ── Metric definitions (name, color, description) ────────────────────────────
METRICS = [
    ("workload",  "#f1a24e", "Mental Workload",       "(θ+α)/(α+β)   frontal"),
    ("fatigue",   "#f14e7a", "Mental Fatigue",         "θ/α            frontal"),
    ("stress",    "#ff5555", "Stress",                 "β/α            frontal"),
    ("attention", "#44ccff", "Attention / Engagement", "β/(θ+α)        frontal"),
    ("faa",       "#88dd66", "Approach / Withdrawal",  "ln(α_AF8)−ln(α_AF7)   50=neutral"),
]

HIGH_THRESHOLD = 75   # % above which the gauge turns red and vertical bands appear


def _update_area_fills(pw, t_rel: np.ndarray, vals: np.ndarray) -> None:
    """
    Drive both fill curves on `pw`:
      base_fill  — always visible, slightly transparent (below-threshold look)
      thresh_fill — solid, only where vals >= HIGH_THRESHOLD (on top of base)
    """
    pw._base_fill.setData(t_rel, vals)
    y_masked = np.where(vals >= HIGH_THRESHOLD, vals, np.nan)
    pw._thresh_fill.setData(t_rel, y_masked)


class _GaugeWidget(QWidget):
    """Vertical QProgressBar with a title, value label, and threshold colour."""

    def __init__(self, color: str, parent=None):
        super().__init__(parent)
        self._base_color = color
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self._bar = QProgressBar()
        self._bar.setOrientation(Qt.Orientation.Vertical)
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setMinimumHeight(140)
        self._bar.setFixedWidth(44)
        self._apply_style(color)

        self._value_lbl = QLabel("0")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_lbl.setStyleSheet(
            f"font-size: 18px; font-weight: bold; color: {color};"
        )

        bar_row = QHBoxLayout()
        bar_row.addStretch()
        bar_row.addWidget(self._bar)
        bar_row.addStretch()
        layout.addLayout(bar_row, stretch=1)
        layout.addWidget(self._value_lbl)

    def set_value(self, value: float):
        v = int(round(max(0, min(100, value))))
        self._bar.setValue(v)
        self._value_lbl.setText(str(v))
        chunk_color = "#ff3333" if v >= HIGH_THRESHOLD else (
            "#f1a24e" if v >= 50 else self._base_color
        )
        self._apply_style(chunk_color)

    def _apply_style(self, color: str):
        self._bar.setStyleSheet(
            f"QProgressBar {{ border: 1px solid #333355; border-radius: 6px; "
            f"background-color: #0a1628; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 4px; }}"
        )


class MentalPanel(QWidget):
    """
    Mental State tab.
    Call refresh(metrics, processor) from the slow QTimer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._gauges: dict[str, _GaugeWidget] = {}
        self._hist_plots: dict[str, pg.PlotWidget] = {}
        self._hist_curves: dict[str, pg.PlotDataItem] = {}
        self._ref_plot: pg.PlotWidget | None = None   # x-axis link reference
        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        # ── Gauge row ─────────────────────────────────────────────────────────
        gauge_scroll = QScrollArea()
        gauge_scroll.setWidgetResizable(True)
        gauge_scroll.setMaximumHeight(200)
        gauge_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        gauge_scroll.setStyleSheet("QScrollArea { border: none; }")

        gauge_container = QWidget()
        gauge_row = QHBoxLayout(gauge_container)
        gauge_row.setSpacing(12)
        gauge_row.addStretch()

        for key, color, title, desc in METRICS:
            col = QVBoxLayout()
            col.setSpacing(2)
            col.setAlignment(Qt.AlignmentFlag.AlignHCenter)

            t_lbl = QLabel(title)
            t_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            t_lbl.setStyleSheet(
                f"font-size: 11px; font-weight: bold; color: {color};"
            )
            d_lbl = QLabel(desc)
            d_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            d_lbl.setStyleSheet("font-size: 9px; color: #666666;")

            gauge = _GaugeWidget(color)
            self._gauges[key] = gauge

            col.addWidget(t_lbl)
            col.addWidget(d_lbl)
            col.addWidget(gauge, alignment=Qt.AlignmentFlag.AlignHCenter)
            gauge_row.addLayout(col)
            gauge_row.addStretch()

        gauge_scroll.setWidget(gauge_container)
        root.addWidget(gauge_scroll)

        # ── Stacked history plots (vertical, linked x-axis) ───────────────────
        vsplit = QSplitter(Qt.Orientation.Vertical)

        for key, color, title, _desc in METRICS:
            pw = self._make_hist_plot(title, color, key == "faa")
            if self._ref_plot is None:
                self._ref_plot = pw
            else:
                pw.setXLink(self._ref_plot)
            self._hist_plots[key] = pw
            vsplit.addWidget(pw)

        # Equal initial sizes
        n = len(METRICS)
        vsplit.setSizes([200] * n)

        root.addWidget(vsplit, stretch=1)

    def _make_hist_plot(self, title: str, color: str, is_faa: bool = False) -> pg.PlotWidget:
        pw = pg.PlotWidget(background="#111111")
        pw.setMenuEnabled(False)
        pw.setMouseEnabled(x=True, y=True)   # navigation enabled
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.setLabel("left", title, color=color)
        pw.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        pw.getAxis("left").setTextPen(pg.mkPen(color))
        pw.getAxis("left").setWidth(80)

        if is_faa:
            pw.setYRange(0, 100, padding=0.05)
            mid = pg.InfiniteLine(
                pos=50, angle=0,
                pen=pg.mkPen("#888888", width=1, style=Qt.PenStyle.DashLine),
                label="Neutral", labelOpts={"color": "#888888", "position": 0.02},
            )
            pw.addItem(mid)
        else:
            pw.setYRange(0, 100, padding=0.05)
            thresh = pg.InfiniteLine(
                pos=HIGH_THRESHOLD, angle=0,
                pen=pg.mkPen("#ff6666", width=1, style=Qt.PenStyle.DashLine),
                label="High", labelOpts={"color": "#ff6666", "position": 0.98},
            )
            pw.addItem(thresh)

        pen = pg.mkPen(color=color, width=1.5)
        curve = pw.plot([], [], pen=pen)

        base_fill = pw.plot(
            [], [],
            pen=pg.mkPen(None),
            fillLevel=0,
            brush=pg.mkBrush(color + "2a"),
            connect="finite",
        )
        base_fill.setZValue(-6)

        thresh_fill = pw.plot(
            [], [],
            pen=pg.mkPen(None),
            fillLevel=0,
            brush=pg.mkBrush(color + "bb"),
            connect="finite",
        )
        thresh_fill.setZValue(-5)

        pw.getAxis("bottom").setStyle(showValues=False)   # hide x labels except bottom
        self._hist_curves[f"_curve_{id(pw)}"] = curve
        pw._metric_curve  = curve
        pw._metric_color  = color
        pw._is_faa        = is_faa
        pw._base_fill     = base_fill
        pw._thresh_fill   = thresh_fill
        return pw

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def refresh(self, metrics: dict, processor):
        """
        metrics   : {"workload":float, "fatigue":float, "stress":float,
                     "attention":float, "faa":float}  — all 0-100
        processor : EEGProcessor (for get_metric_history)
        """
        for key, _color, _title, _desc in METRICS:
            val = metrics.get(key, 0.0)
            self._gauges[key].set_value(val)

            pw = self._hist_plots[key]
            ts, vals = processor.get_metric_history(key)
            if len(ts) < 2:
                continue
            t_rel = ts - ts[0]
            pw._metric_curve.setData(t_rel, vals)

            # Fill: subtle always-on base + solid overlay where threshold exceeded
            if not pw._is_faa:
                _update_area_fills(pw, t_rel, vals)

        # Show x-axis labels only on the bottom-most plot
        for key, _c, _t, _d in METRICS:
            pw = self._hist_plots[key]
            is_last = (key == METRICS[-1][0])
            pw.getAxis("bottom").setStyle(showValues=is_last)
            if is_last:
                pw.setLabel("bottom", "Time (s)", color="#888")
