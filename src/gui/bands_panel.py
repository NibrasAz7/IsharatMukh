"""
bands_panel.py
EEG Bands tab — grouped bar chart (current power) + power history,
with per-channel toggle checkboxes.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QCheckBox, QHBoxLayout, QLabel, QSplitter, QVBoxLayout, QWidget,
)

from src.gui.theme import BAND_COLORS, CH_NAMES, CHANNEL_COLORS


_BANDS = ["delta", "theta", "alpha", "beta", "gamma"]
_N_CH  = 4   # TP9, AF7, AF8, TP10 (no AUX for bands)

_BAR_WIDTH  = 0.15
_BAR_GAP    = 0.05


class BandsPanel(QWidget):
    """
    EEG Bands tab.
    Call update(powers, band_history) from the slow QTimer (0.5 Hz).
    powers       : dict band→ndarray(n_channels)
    band_history : dict band→deque of (timestamp, avg_power)
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ch_visible = [True] * _N_CH
        self._setup_ui()

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Channel toggles ───────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.setSpacing(16)
        toggle_row.addWidget(QLabel("Channels:"))
        self._toggles: list[QCheckBox] = []
        for i, ch in enumerate(CH_NAMES[:_N_CH]):
            cb = QCheckBox(ch)
            cb.setChecked(True)
            color = CHANNEL_COLORS[ch]
            cb.setStyleSheet(
                f"QCheckBox {{ color: {color}; font-weight: bold; }}"
                f"QCheckBox::indicator:checked {{ background-color: {color}; border-color: {color}; }}"
            )
            cb.stateChanged.connect(lambda state, idx=i: self._on_toggle(idx, state))
            self._toggles.append(cb)
            toggle_row.addWidget(cb)
        toggle_row.addStretch()
        root.addLayout(toggle_row)

        # ── Splitter: bar chart top, history bottom ───────────────────────────
        splitter = QSplitter(Qt.Orientation.Vertical)

        # Bar chart
        self._bar_plot = pg.PlotWidget(background="#111111")
        self._bar_plot.setMouseEnabled(x=False, y=False)
        self._bar_plot.setMenuEnabled(False)
        self._bar_plot.showGrid(x=False, y=True, alpha=0.2)
        self._bar_plot.setLabel("left", "µV²", color="#888")
        self._bar_plot.getAxis("left").setTextPen(pg.mkPen("#888"))

        # X-axis labels = band names
        ax = self._bar_plot.getAxis("bottom")
        ax.setTextPen(pg.mkPen("#888"))
        x_ticks = [(i + _BAR_WIDTH * _N_CH / 2, b) for i, b in enumerate(_BANDS)]
        ax.setTicks([x_ticks])

        # Pre-create one BarGraphItem per (band × channel)
        self._bars: list[list[pg.BarGraphItem]] = []
        for b_idx, band in enumerate(_BANDS):
            ch_bars = []
            for c_idx in range(_N_CH):
                x = b_idx + c_idx * (_BAR_WIDTH + _BAR_GAP / _N_CH)
                color = QColor(CHANNEL_COLORS[CH_NAMES[c_idx]])
                bar = pg.BarGraphItem(x=[x], height=[0], width=_BAR_WIDTH,
                                      brush=color, pen=pg.mkPen(None))
                self._bar_plot.addItem(bar)
                ch_bars.append(bar)
            self._bars.append(ch_bars)

        # History plot
        self._hist_plot = pg.PlotWidget(background="#111111")
        self._hist_plot.setMouseEnabled(x=False, y=False)
        self._hist_plot.setMenuEnabled(False)
        self._hist_plot.showGrid(x=True, y=True, alpha=0.2)
        self._hist_plot.setLabel("bottom", "Time (s)", color="#888")
        self._hist_plot.setLabel("left", "µV²", color="#888")
        self._hist_plot.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        self._hist_plot.getAxis("left").setTextPen(pg.mkPen("#888"))
        self._hist_curves: dict[str, pg.PlotDataItem] = {}
        for band in _BANDS:
            pen = pg.mkPen(color=BAND_COLORS[band], width=1.5)
            curve = self._hist_plot.plot([], [], pen=pen, name=band)
            self._hist_curves[band] = curve

        legend = self._hist_plot.addLegend(offset=(10, 10))
        legend.setLabelTextColor("#ccc")

        splitter.addWidget(self._bar_plot)
        splitter.addWidget(self._hist_plot)
        splitter.setSizes([350, 250])
        root.addWidget(splitter, stretch=1)

    # ── Public API ────────────────────────────────────────────────────────────

    def refresh(self, powers: dict, band_history: dict):
        """
        powers       : dict  band_name → ndarray shape (≥4,)  µV²
        band_history : dict  band_name → deque of (timestamp, avg_power)
        """
        # Update bar heights
        for b_idx, band in enumerate(_BANDS):
            band_pow = powers.get(band, np.zeros(_N_CH))
            for c_idx in range(_N_CH):
                h = float(band_pow[c_idx]) if len(band_pow) > c_idx else 0.0
                visible = self._ch_visible[c_idx]
                self._bars[b_idx][c_idx].setOpts(height=h if visible else 0)

        # Update history curves
        for band, hist in band_history.items():
            if not hist or band not in self._hist_curves:
                continue
            ts  = [h[0] for h in hist]
            val = [h[1] for h in hist]
            t_rel = [t - ts[0] for t in ts]
            self._hist_curves[band].setData(t_rel, val)

    # ── Private ───────────────────────────────────────────────────────────────

    def _on_toggle(self, ch_idx: int, state: int):
        self._ch_visible[ch_idx] = (state == Qt.CheckState.Checked.value)
        # Re-render with last data if needed; bars will be updated on next tick
        for b_idx in range(len(_BANDS)):
            bar = self._bars[b_idx][ch_idx]
            if not self._ch_visible[ch_idx]:
                bar.setOpts(height=0)
