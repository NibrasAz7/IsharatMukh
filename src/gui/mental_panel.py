"""
mental_panel.py
Mental State tab — Workload + Fatigue gauges (QProgressBar) + history plots.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QSplitter, QVBoxLayout, QWidget,
)


class _GaugeWidget(QWidget):
    """Labelled QProgressBar styled as a vertical gauge."""

    def __init__(self, title: str, color: str, parent=None):
        super().__init__(parent)
        self._color = color

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 8, 16, 8)
        layout.setSpacing(4)

        title_lbl = QLabel(title)
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_lbl.setStyleSheet("font-weight: bold; font-size: 15px; color: #ffffff;")

        self._bar = QProgressBar()
        self._bar.setOrientation(Qt.Orientation.Vertical)
        self._bar.setMinimum(0)
        self._bar.setMaximum(100)
        self._bar.setValue(0)
        self._bar.setTextVisible(False)
        self._bar.setMinimumHeight(200)
        self._bar.setFixedWidth(60)
        self._bar.setStyleSheet(
            f"QProgressBar {{ border: 2px solid #333355; border-radius: 8px; "
            f"background-color: #0a1628; }}"
            f"QProgressBar::chunk {{ background-color: {color}; border-radius: 6px; }}"
        )

        self._value_lbl = QLabel("0 %")
        self._value_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._value_lbl.setStyleSheet(
            f"font-size: 22px; font-weight: bold; color: {color};"
        )

        bar_row = QHBoxLayout()
        bar_row.addStretch()
        bar_row.addWidget(self._bar)
        bar_row.addStretch()

        layout.addWidget(title_lbl)
        layout.addLayout(bar_row, stretch=1)
        layout.addWidget(self._value_lbl)

    def set_value(self, value: float):
        v = int(round(max(0, min(100, value))))
        self._bar.setValue(v)
        self._value_lbl.setText(f"{v} %")
        # Change colour based on level
        if v >= 75:
            chunk_color = "#ff4444"
        elif v >= 50:
            chunk_color = "#f1a24e"
        else:
            chunk_color = self._color
        self._bar.setStyleSheet(
            f"QProgressBar {{ border: 2px solid #333355; border-radius: 8px; "
            f"background-color: #0a1628; }}"
            f"QProgressBar::chunk {{ background-color: {chunk_color}; border-radius: 6px; }}"
        )


class MentalPanel(QWidget):
    """
    Mental State tab.
    Call update(metrics, processor) from the slow QTimer (0.5 Hz).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(8)

        # ── Gauge row ─────────────────────────────────────────────────────────
        gauge_row = QHBoxLayout()
        gauge_row.setSpacing(32)

        wl_col = QVBoxLayout()
        wl_col.setSpacing(4)
        wl_title = QLabel("Mental Workload")
        wl_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f1a24e;")
        wl_desc = QLabel("Frontal θ+α / α+β  —  rises with cognitive effort")
        wl_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        wl_desc.setStyleSheet("font-size: 11px; color: #888888;")
        self._wl_gauge = _GaugeWidget("", "#f1a24e")
        wl_col.addWidget(wl_title)
        wl_col.addWidget(wl_desc)
        wl_col.addWidget(self._wl_gauge)

        fa_col = QVBoxLayout()
        fa_col.setSpacing(4)
        fa_title = QLabel("Mental Fatigue")
        fa_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fa_title.setStyleSheet("font-size: 15px; font-weight: bold; color: #f14e7a;")
        fa_desc = QLabel("Frontal θ / α  —  rises with sustained mental effort")
        fa_desc.setAlignment(Qt.AlignmentFlag.AlignCenter)
        fa_desc.setStyleSheet("font-size: 11px; color: #888888;")
        self._fa_gauge = _GaugeWidget("", "#f14e7a")
        fa_col.addWidget(fa_title)
        fa_col.addWidget(fa_desc)
        fa_col.addWidget(self._fa_gauge)

        gauge_row.addStretch()
        gauge_row.addLayout(wl_col)
        gauge_row.addStretch()
        gauge_row.addLayout(fa_col)
        gauge_row.addStretch()
        root.addLayout(gauge_row)

        # ── History plots ─────────────────────────────────────────────────────
        hist_row = QHBoxLayout()
        hist_row.setSpacing(8)

        self._wl_hist = self._make_hist_plot("Mental Workload over time", "#f1a24e")
        self._fa_hist = self._make_hist_plot("Mental Fatigue over time",  "#f14e7a")

        hist_row.addWidget(self._wl_hist, stretch=1)
        hist_row.addWidget(self._fa_hist, stretch=1)
        root.addLayout(hist_row, stretch=1)

    @staticmethod
    def _make_hist_plot(title: str, color: str) -> pg.PlotWidget:
        pw = pg.PlotWidget(background="#111111", title=f"<span style='color:#ccc'>{title}</span>")
        pw.setMouseEnabled(x=False, y=False)
        pw.setMenuEnabled(False)
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.setLabel("bottom", "Time (s)", color="#888")
        pw.setLabel("left", "%", color="#888")
        pw.setYRange(0, 100, padding=0.05)
        pw.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        pw.getAxis("left").setTextPen(pg.mkPen("#888"))
        # Add 75 % threshold line
        threshold = pg.InfiniteLine(
            pos=75, angle=0,
            pen=pg.mkPen(color="#ff6666", width=1, style=Qt.PenStyle.DashLine),
            label="High", labelOpts={"color": "#ff6666", "position": 0.95},
        )
        pw.addItem(threshold)
        # Curve
        pen = pg.mkPen(color=color, width=2)
        fill = pg.mkBrush(color + "33")
        curve = pw.plot([], [], pen=pen, fillLevel=0, brush=fill, name=title)
        pw._eeg_curve = curve   # store reference for update
        return pw

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self, metrics: dict, processor):
        """
        metrics   : {"workload": float 0-100, "fatigue": float 0-100}
        processor : EEGProcessor (for get_metric_history)
        """
        wl = metrics.get("workload", 0.0)
        fa = metrics.get("fatigue",  0.0)

        self._wl_gauge.set_value(wl)
        self._fa_gauge.set_value(fa)

        # History
        for metric, pw in (("workload", self._wl_hist), ("fatigue", self._fa_hist)):
            ts, vals = processor.get_metric_history(metric)
            if len(ts) > 0:
                t_rel = ts - ts[0]
                pw._eeg_curve.setData(t_rel, vals)
