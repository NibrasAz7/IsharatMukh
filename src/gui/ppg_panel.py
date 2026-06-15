"""
ppg_panel.py
PPG (photoplethysmography) tab — raw sweep + HR + HRV gauges/history.
"""

import numpy as np
import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QHBoxLayout, QLabel, QProgressBar, QSplitter, QVBoxLayout, QWidget,
)

try:
    from scipy.signal import find_peaks
    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

PPG_CH_NAMES  = ["Ambient", "IR", "Red"]
PPG_CH_COLORS = {"Ambient": "#888888", "IR": "#ff5555", "Red": "#ff2222"}
PPG_WINDOW_S  = 10       # seconds shown in sweep
PPG_FS        = 64       # Hz
PPG_SWEEP_PTS = PPG_FS * PPG_WINDOW_S   # 640 samples


class _NumericGauge(QWidget):
    """Simple labelled numeric readout (not a bar)."""

    def __init__(self, title: str, unit: str, color: str, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(2)
        layout.setAlignment(Qt.AlignmentFlag.AlignHCenter)

        t = QLabel(title)
        t.setAlignment(Qt.AlignmentFlag.AlignCenter)
        t.setStyleSheet(f"font-size: 12px; font-weight: bold; color: {color};")

        self._val_lbl = QLabel("—")
        self._val_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._val_lbl.setStyleSheet(
            f"font-size: 36px; font-weight: bold; color: {color};"
        )
        u = QLabel(unit)
        u.setAlignment(Qt.AlignmentFlag.AlignCenter)
        u.setStyleSheet("font-size: 11px; color: #666;")

        layout.addWidget(t)
        layout.addWidget(self._val_lbl)
        layout.addWidget(u)

    def set_value(self, v: float | None):
        if v is None or np.isnan(v):
            self._val_lbl.setText("—")
        else:
            self._val_lbl.setText(f"{v:.0f}")


class PPGPanel(QWidget):
    """
    PPG tab panel.
    Call refresh(ppg_window) from the PPG fast timer (10 Hz).
    Call refresh_metrics(hr, hrv) from the slow timer (2 s).
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ch_visible = {ch: True for ch in PPG_CH_NAMES}
        self._x_arr = np.linspace(0, PPG_WINDOW_S, PPG_SWEEP_PTS)
        self._nan_y = np.full(PPG_SWEEP_PTS, np.nan)
        # HR/HRV history as lists of (t_rel, value)
        self._hr_hist:  list[tuple[float, float]] = []
        self._hrv_hist: list[tuple[float, float]] = []
        self._t0 = 0.0
        self._setup_ui()

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _setup_ui(self):
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── Toggle row ────────────────────────────────────────────────────────
        toggle_row = QHBoxLayout()
        toggle_row.addWidget(QLabel("Channels:"))
        for ch in PPG_CH_NAMES:
            from PyQt6.QtWidgets import QCheckBox
            cb = QCheckBox(ch)
            cb.blockSignals(True)
            cb.setChecked(True)
            cb.blockSignals(False)
            cb.setStyleSheet(
                f"QCheckBox {{ color: {PPG_CH_COLORS[ch]}; font-weight: bold; }}"
            )
            cb.stateChanged.connect(
                lambda state, c=ch: self._on_toggle(c, state)
            )
            toggle_row.addWidget(cb)
        toggle_row.addStretch()
        root.addLayout(toggle_row)

        # ── Raw sweep ─────────────────────────────────────────────────────────
        self._sweep = pg.PlotWidget(background="#111111")
        self._sweep.setMouseEnabled(x=False, y=True)
        self._sweep.showGrid(x=True, y=True, alpha=0.2)
        self._sweep.setLabel("bottom", "Time (s)", color="#888")
        self._sweep.setLabel("left", "PPG (a.u.)", color="#888")
        self._sweep.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        self._sweep.getAxis("left").setTextPen(pg.mkPen("#888"))
        self._sweep.setXRange(0, PPG_WINDOW_S, padding=0)
        self._sweep.setMenuEnabled(False)
        self._sweep_curves: dict[str, pg.PlotDataItem] = {}
        for ch in PPG_CH_NAMES:
            pen = pg.mkPen(color=PPG_CH_COLORS[ch], width=1.5)
            curve = self._sweep.plot(self._x_arr, self._nan_y.copy(),
                                     pen=pen, name=ch, connect='finite')
            self._sweep_curves[ch] = curve
        root.addWidget(self._sweep, stretch=2)

        # ── HR + HRV row ──────────────────────────────────────────────────────
        metrics_row = QHBoxLayout()
        metrics_row.setSpacing(20)

        self._hr_gauge  = _NumericGauge("Heart Rate",  "bpm",  "#ff5555")
        self._hrv_gauge = _NumericGauge("HRV (SDNN)", "ms",   "#55aaff")

        metrics_row.addStretch()
        metrics_row.addWidget(self._hr_gauge)
        metrics_row.addStretch()
        metrics_row.addWidget(self._hrv_gauge)
        metrics_row.addStretch()
        root.addLayout(metrics_row)

        # ── HR / HRV history (linked x) ───────────────────────────────────────
        vsplit = QSplitter(Qt.Orientation.Vertical)

        self._hr_plot  = self._make_hist_plot("Heart Rate (bpm)",  "#ff5555", y_range=(40, 180))
        self._hrv_plot = self._make_hist_plot("HRV SDNN (ms)",     "#55aaff", y_range=(0, 120))
        self._hrv_plot.setXLink(self._hr_plot)

        vsplit.addWidget(self._hr_plot)
        vsplit.addWidget(self._hrv_plot)
        vsplit.setSizes([150, 150])
        root.addWidget(vsplit, stretch=1)

    @staticmethod
    def _make_hist_plot(title: str, color: str, y_range=(0, 100)) -> pg.PlotWidget:
        pw = pg.PlotWidget(background="#111111")
        pw.setMenuEnabled(False)
        pw.setMouseEnabled(x=True, y=True)
        pw.showGrid(x=True, y=True, alpha=0.2)
        pw.setLabel("left", title, color=color)
        pw.setYRange(*y_range, padding=0.05)
        pw.getAxis("bottom").setTextPen(pg.mkPen("#888"))
        pw.getAxis("left").setTextPen(pg.mkPen(color))
        pen  = pg.mkPen(color=color, width=1.5)
        fill = pg.mkBrush(color + "33")
        curve = pw.plot([], [], pen=pen, fillLevel=0, brush=fill, connect='finite')
        pw._ppg_curve = curve
        return pw

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def refresh_sweep(self, ppg_window: np.ndarray):
        """
        ppg_window : (n_samples, 3)  — latest N seconds of PPG (ambient, IR, red)
        n_samples  ≤ PPG_SWEEP_PTS
        """
        n = ppg_window.shape[0]
        for ci, ch in enumerate(PPG_CH_NAMES):
            if not self._ch_visible[ch]:
                self._sweep_curves[ch].setData(self._x_arr, self._nan_y)
                continue
            y = np.full(PPG_SWEEP_PTS, np.nan)
            if n > 0:
                start = max(0, PPG_SWEEP_PTS - n)
                y[start:] = ppg_window[-PPG_SWEEP_PTS + start:, ci] if n >= PPG_SWEEP_PTS else ppg_window[:, ci]
            self._sweep_curves[ch].setData(self._x_arr, y, connect='finite')

    def refresh_metrics(self, hr: float | None, hrv: float | None, t_rel: float):
        """Update gauges and history with latest HR/HRV values."""
        self._hr_gauge.set_value(hr)
        self._hrv_gauge.set_value(hrv)

        if hr is not None and not np.isnan(hr):
            self._hr_hist.append((t_rel, hr))
        if hrv is not None and not np.isnan(hrv):
            self._hrv_hist.append((t_rel, hrv))

        # Keep last 10 min
        cutoff = t_rel - 600
        self._hr_hist  = [(t, v) for t, v in self._hr_hist  if t >= cutoff]
        self._hrv_hist = [(t, v) for t, v in self._hrv_hist if t >= cutoff]

        if len(self._hr_hist) >= 2:
            ts, vals = zip(*self._hr_hist)
            self._hr_plot._ppg_curve.setData(list(ts), list(vals))
        if len(self._hrv_hist) >= 2:
            ts, vals = zip(*self._hrv_hist)
            self._hrv_plot._ppg_curve.setData(list(ts), list(vals))

    def reset(self):
        """Blank the sweep on disconnect."""
        for curve in self._sweep_curves.values():
            curve.setData(self._x_arr, self._nan_y)
        self._hr_gauge.set_value(None)
        self._hrv_gauge.set_value(None)
        self._hr_hist.clear()
        self._hrv_hist.clear()

    # ─────────────────────────────────────────────────────────────────────────
    # Private
    # ─────────────────────────────────────────────────────────────────────────

    def _on_toggle(self, ch: str, state: int):
        from PyQt6.QtCore import Qt
        self._ch_visible[ch] = (state == Qt.CheckState.Checked.value)


# ── HR / HRV computation helper (pure function) ──────────────────────────────

def compute_hr_hrv(ir_signal: np.ndarray, fs: float = PPG_FS
                   ) -> tuple[float | None, float | None]:
    """
    Estimate HR (bpm) and HRV SDNN (ms) from the IR PPG channel.
    Returns (None, None) if insufficient data or no peaks found.
    """
    if not _HAS_SCIPY or ir_signal.shape[0] < int(fs * 5):
        return None, None
    try:
        # Normalise
        sig = ir_signal - np.mean(ir_signal)
        sig_range = np.ptp(sig)
        if sig_range < 1e-6:
            return None, None
        sig = sig / sig_range

        min_dist = int(fs * 0.4)   # 0.4 s ≈ 150 bpm max
        peaks, _ = find_peaks(sig, height=0.2, distance=min_dist)
        if len(peaks) < 3:
            return None, None

        rr = np.diff(peaks) / fs * 1000   # ms
        hr  = 60_000.0 / np.mean(rr)
        hrv = float(np.std(rr))
        if not (30 < hr < 220):
            return None, None
        return float(hr), hrv
    except Exception:
        return None, None
