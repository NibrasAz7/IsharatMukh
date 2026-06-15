"""
eeg_panel.py
Raw EEG sweep panel — ECG-style left→right scrolling display.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QVBoxLayout, QWidget,
)

from src.gui.theme import (
    CH_NAMES, CH_OFFSETS, CH_YRANGE, CHANNEL_COLORS,
    QUALITY_COLORS, SWEEP_SIZE, SWEEP_WINDOW_S,
)


class EEGPanel(QWidget):
    """
    Raw EEG sweep tab.
    Call update(y_sweep, quality) from the main window's fast timer.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._x_arr = np.linspace(0, SWEEP_WINDOW_S, SWEEP_SIZE)
        self._nan_y = np.full(SWEEP_SIZE, np.nan)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Channel toggle row ────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(16)
        self._toggles: dict[str, QCheckBox] = {}
        for i, ch in enumerate(CH_NAMES):
            cb = QCheckBox(ch)
            cb.setChecked(i < 4)   # AUX off by default
            color = CHANNEL_COLORS[ch]
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: bold; }}"
                f"QCheckBox::indicator:checked {{ background-color: {color}; border-color: {color}; }}"
            )
            cb.stateChanged.connect(lambda state, idx=i: self._on_toggle(idx, state))
            self._toggles[ch] = cb
            toggle_row.addWidget(cb)
        toggle_row.addStretch()

        # ── Quality indicators ────────────────────────────────────────────────
        quality_row = QHBoxLayout()
        quality_row.setSpacing(8)
        quality_row.addWidget(QLabel("Signal:"))
        self._quality_labels: list[QLabel] = []
        for ch in CH_NAMES[:4]:
            lbl = QLabel(ch)
            lbl.setObjectName("status-label")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedWidth(50)
            lbl.setStyleSheet(
                f"background-color: {QUALITY_COLORS['secondary']}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )
            self._quality_labels.append(lbl)
            quality_row.addWidget(lbl)
        quality_row.addStretch()

        top_row = QHBoxLayout()
        top_row.addLayout(toggle_row)
        top_row.addLayout(quality_row)
        root.addLayout(top_row)

        # ── Sweep plot ────────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(background="#111111")
        self._plot.setMouseEnabled(x=False, y=False)
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time (s)", color="#888")
        self._plot.setLabel("left", "µV (offset per channel)", color="#888")
        self._plot.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        self._plot.getAxis("left").setTextPen(pg.mkPen("#888"))
        self._plot.setXRange(0, SWEEP_WINDOW_S, padding=0)
        self._plot.setYRange(CH_YRANGE[0], CH_YRANGE[1], padding=0)
        self._plot.setMenuEnabled(False)

        # Tick marks at y-offsets with channel names
        y_axis = self._plot.getAxis("left")
        y_axis.setTicks([[(off, name) for name, off in zip(CH_NAMES, CH_OFFSETS)]])

        # One curve per channel
        self._curves: list[pg.PlotDataItem] = []
        for i, ch in enumerate(CH_NAMES):
            pen = pg.mkPen(color=CHANNEL_COLORS[ch], width=1)
            curve = self._plot.plot(self._x_arr, self._nan_y.copy(), pen=pen, name=ch)
            curve.setVisible(i < 4)
            self._curves.append(curve)

        root.addWidget(self._plot, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def update(self, y_sweep: np.ndarray, quality: list[str]):
        """
        Called by the fast QTimer (10 Hz).
        y_sweep : (SWEEP_SIZE, 5) — raw µV, NaN where no data / eraser gap
        quality : list of 4 labels ('good'/'fair'/'poor')
        """
        for i in range(len(CH_NAMES)):
            if not self._curves[i].isVisible():
                continue
            y = y_sweep[:, i].copy()
            mask = ~np.isnan(y)
            y[mask] += CH_OFFSETS[i]
            self._curves[i].setData(self._x_arr, y)

        for i, (lbl, q) in enumerate(zip(self._quality_labels, quality)):
            color = QUALITY_COLORS.get(q, QUALITY_COLORS["secondary"])
            lbl.setStyleSheet(
                f"background-color: {color}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )

    def reset(self):
        """Blank all curves (called on disconnect)."""
        nan_y = self._nan_y.copy()
        for curve in self._curves:
            curve.setData(self._x_arr, nan_y)
        for lbl in self._quality_labels:
            lbl.setStyleSheet(
                f"background-color: {QUALITY_COLORS['secondary']}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_toggle(self, ch_idx: int, state: int):
        visible = state == Qt.CheckState.Checked.value
        self._curves[ch_idx].setVisible(visible)
