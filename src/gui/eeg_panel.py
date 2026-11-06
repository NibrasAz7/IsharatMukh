"""
eeg_panel.py
Raw EEG sweep panel — ECG-style left→right scrolling display.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QHBoxLayout, QLabel, QPushButton, QVBoxLayout, QWidget,
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
        self._sim_phase    = 0.0
        self._sim_active  = False
        self._display_mode = 'scroll'   # 'scroll' | 'sweep'
        self._sim_timer   = QTimer(self)
        self._sim_timer.setInterval(100)
        self._sim_timer.timeout.connect(self._sim_tick)
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)
        self._curves: list[pg.PlotDataItem] = []  # init early so _on_toggle is safe

        # ── Channel toggle row ────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(16)
        self._toggles: dict[str, QCheckBox] = {}
        for i, ch in enumerate(CH_NAMES):
            cb = QCheckBox(ch)
            cb.blockSignals(True)
            cb.setChecked(i < 4)   # AUX off by default
            cb.blockSignals(False)
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
            lbl = QLabel(f"{ch}\n—")
            lbl.setObjectName("status-label")
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setFixedWidth(58)
            lbl.setToolTip(
                "Contact-quality estimate (µV RMS of filtered signal).\n"
                "Proxy for electrode contact — true impedance is not "
                "exposed by the Muse S over BLE."
            )
            lbl.setStyleSheet(
                f"background-color: {QUALITY_COLORS['secondary']}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )
            self._quality_labels.append(lbl)
            quality_row.addWidget(lbl)
        quality_row.addStretch()

        self._sim_btn = QPushButton("▶ Simulate")
        self._sim_btn.setCheckable(True)
        self._sim_btn.setFixedWidth(100)
        self._sim_btn.setToolTip("Generate synthetic sine-wave EEG without hardware")
        self._sim_btn.clicked.connect(self._on_simulate_toggle)
        quality_row.addWidget(self._sim_btn)

        quality_row.addWidget(QLabel("Mode:"))
        self._mode_combo = QComboBox()
        self._mode_combo.addItems(["Scrolling", "Sweep"])
        self._mode_combo.setFixedWidth(90)
        self._mode_combo.setToolTip(
            "Scrolling: latest data always on the right.\n"
            "Sweep: draw-and-erase ECG-style cursor."
        )
        self._mode_combo.currentIndexChanged.connect(self._on_mode_change)
        quality_row.addWidget(self._mode_combo)

        top_row = QHBoxLayout()
        top_row.addLayout(toggle_row)
        top_row.addLayout(quality_row)
        root.addLayout(top_row)

        # ── Sweep plot ────────────────────────────────────────────────────────
        self._plot = pg.PlotWidget(background="#111111")
        self._plot.setMouseEnabled(x=True, y=False)   # allow x pan/zoom; y is locked
        self._plot.showGrid(x=True, y=True, alpha=0.2)
        self._plot.setLabel("bottom", "Time (s)", color="#888")
        self._plot.setLabel("left", "µV (offset per channel)", color="#888")
        self._plot.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        self._plot.getAxis("left").setTextPen(pg.mkPen("#888"))
        self._plot.setXRange(0, SWEEP_WINDOW_S, padding=0)
        self._plot.setYRange(CH_YRANGE[0], CH_YRANGE[1], padding=0)
        self._plot.disableAutoRange()
        self._plot.setMenuEnabled(False)

        # Tick marks at y-offsets with channel names
        y_axis = self._plot.getAxis("left")
        y_axis.setTicks([[(off, name) for name, off in zip(CH_NAMES, CH_OFFSETS)]])

        # One curve per channel — connect='finite' treats NaN as gap
        for i, ch in enumerate(CH_NAMES):
            pen = pg.mkPen(color=CHANNEL_COLORS[ch], width=1)
            curve = self._plot.plot(self._x_arr, self._nan_y.copy(),
                                    pen=pen, name=ch, connect='finite')
            curve.setVisible(i < 4)
            self._curves.append(curve)

        root.addWidget(self._plot, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    @property
    def display_mode(self) -> str:
        """'scroll' or 'sweep'."""
        return self._display_mode

    def refresh(self, y_sweep: np.ndarray, stats: list[dict]):
        """
        Called by the fast QTimer (10 Hz).
        y_sweep : (SWEEP_SIZE, 5) — raw µV, NaN where no data / eraser gap
        stats   : list of 4 dicts {"rms", "line_noise", "label"}
        """
        if self._sim_active:
            return   # simulation drives the curves directly
        for i in range(len(CH_NAMES)):
            if not self._curves[i].isVisible():
                continue
            y = y_sweep[:, i].copy()
            mask = ~np.isnan(y)
            y[mask] += CH_OFFSETS[i]
            self._curves[i].setData(self._x_arr, y, connect='finite')

        for i, (lbl, s) in enumerate(zip(self._quality_labels, stats)):
            color = QUALITY_COLORS.get(s.get("label", "poor"),
                                       QUALITY_COLORS["secondary"])
            lbl.setText(f"{CH_NAMES[i]}\n{s.get('rms', 0.0):.0f} µV")
            lbl.setStyleSheet(
                f"background-color: {color}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )

    def reset(self):
        """Blank all curves (called on disconnect)."""
        nan_y = self._nan_y.copy()
        for curve in self._curves:
            curve.setData(self._x_arr, nan_y)
        self._plot.disableAutoRange()
        for i, lbl in enumerate(self._quality_labels):
            lbl.setText(f"{CH_NAMES[i]}\n—")
            lbl.setStyleSheet(
                f"background-color: {QUALITY_COLORS['secondary']}; "
                "border-radius: 4px; padding: 2px 4px; color: white; font-weight: bold;"
            )

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_toggle(self, ch_idx: int, state: int):
        visible = state == Qt.CheckState.Checked.value
        self._curves[ch_idx].setVisible(visible)

    def _on_mode_change(self, index: int):
        self._display_mode = 'scroll' if index == 0 else 'sweep'

    def _on_simulate_toggle(self, checked: bool):
        self._sim_active = checked
        self._sim_btn.setText("⏹ Stop Sim" if checked else "▶ Simulate")
        if checked:
            self._sim_phase = 0.0
            self._sim_timer.start()
        else:
            self._sim_timer.stop()
            self.reset()

    def _sim_tick(self):
        """Generate synthetic multi-frequency EEG to verify rendering pipeline."""
        dt = 0.1  # 100 ms per tick
        self._sim_phase += dt
        t = self._x_arr
        freqs  = [10.0, 8.0, 12.0, 9.0, 7.0]
        amps   = [40.0, 35.0, 30.0, 38.0, 20.0]
        for curve, f, a, off in zip(self._curves, freqs, amps, CH_OFFSETS):
            if not curve.isVisible():
                continue
            y = a * np.sin(2 * np.pi * f * (t + self._sim_phase)) + off
            curve.setData(self._x_arr, y)
