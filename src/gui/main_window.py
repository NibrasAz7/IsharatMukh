"""
main_window.py
QMainWindow for the Muse S EEG Desktop Dashboard.
"""

import time
import traceback
from collections import deque

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox, QGroupBox, QHBoxLayout, QLabel, QMainWindow,
    QPushButton, QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from src.gui.bands_panel import BandsPanel
from src.gui.eeg_panel import EEGPanel
from src.gui.mental_panel import MentalPanel
from src.gui.theme import (
    CH_NAMES, ERASER_WIDTH, QUALITY_COLORS, SWEEP_SIZE,
)
from src.processing import BANDS, EEGProcessor
from src.stream_reader import StreamReader

_N_CH = 5


class MainWindow(QMainWindow):

    _scan_done = pyqtSignal(list)   # emitted from BLE thread, handled in main thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Muse S EEG Dashboard")
        self.resize(1400, 820)

        # ── Backend singletons ────────────────────────────────────────────────
        self._reader    = StreamReader()
        self._processor = EEGProcessor(notch_freq=50.0)

        # ── Sweep state ───────────────────────────────────────────────────────
        self._y_sweep       = np.full((SWEEP_SIZE, _N_CH), np.nan)
        self._write_idx     = 0
        self._last_seen     = 0
        self._filter_states = self._processor.make_filter_states(_N_CH)

        # ── Band history ──────────────────────────────────────────────────────
        self._band_history: dict[str, deque] = {b: deque(maxlen=3600) for b in BANDS}

        # ── Recording state ───────────────────────────────────────────────────
        self._is_recording  = False
        self._record_start  = 0.0

        self._scan_done.connect(self._apply_scan_results)
        self._build_ui()
        self._build_timers()

    # ═════════════════════════════════════════════════════════════════════════
    # UI construction
    # ═════════════════════════════════════════════════════════════════════════

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root_h = QHBoxLayout(central)
        root_h.setContentsMargins(4, 4, 4, 4)
        root_h.setSpacing(4)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_sidebar())

        # Main tab area
        self._tabs = QTabWidget()
        self._eeg_panel    = EEGPanel()
        self._bands_panel  = BandsPanel()
        self._mental_panel = MentalPanel()
        self._tabs.addTab(self._eeg_panel,    "📈  Raw EEG")
        self._tabs.addTab(self._bands_panel,  "🌊  EEG Bands")
        self._tabs.addTab(self._mental_panel, "🧠  Mental State")
        splitter.addWidget(self._tabs)

        splitter.setSizes([220, 1180])
        splitter.setCollapsible(0, False)
        root_h.addWidget(splitter)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setFixedWidth(220)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Title
        title = QLabel("🧠 Muse S EEG")
        title.setObjectName("title-label")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        # Connection badge + timer row
        badge_row = QHBoxLayout()
        self._conn_badge = QLabel("● OFFLINE")
        self._conn_badge.setObjectName("status-label")
        self._conn_badge.setStyleSheet(
            f"background-color: {QUALITY_COLORS['secondary']}; "
            "color: white; border-radius: 4px; padding: 2px 6px; font-weight: bold;"
        )
        self._timer_lbl = QLabel("00:00:00")
        self._timer_lbl.setObjectName("timer-label")
        self._timer_lbl.setStyleSheet("font-family: monospace; color: #aaaaaa;")
        badge_row.addWidget(self._conn_badge)
        badge_row.addStretch()
        badge_row.addWidget(self._timer_lbl)
        layout.addLayout(badge_row)

        # ── Device group ──────────────────────────────────────────────────────
        dev_group = QGroupBox("Device")
        dev_layout = QVBoxLayout(dev_group)
        dev_layout.setSpacing(6)

        self._btn_scan = QPushButton("🔍 Scan for Muse")
        self._btn_scan.clicked.connect(self._on_scan)
        dev_layout.addWidget(self._btn_scan)

        self._scan_status = QLabel("")
        self._scan_status.setStyleSheet("color: #88aaff; font-size: 11px;")
        self._scan_status.setWordWrap(True)
        dev_layout.addWidget(self._scan_status)

        self._device_combo = QComboBox()
        self._device_combo.setPlaceholderText("Select device…")
        dev_layout.addWidget(self._device_combo)

        self._btn_connect = QPushButton("▶  Connect")
        self._btn_connect.setObjectName("btn-connect")
        self._btn_connect.setFixedHeight(44)
        self._btn_connect.clicked.connect(self._on_connect_toggle)
        dev_layout.addWidget(self._btn_connect)

        layout.addWidget(dev_group)

        # ── Recording group ───────────────────────────────────────────────────
        rec_group = QGroupBox("Recording")
        rec_layout = QVBoxLayout(rec_group)
        rec_layout.setSpacing(6)

        self._btn_record = QPushButton("⏺  Start Recording")
        self._btn_record.setObjectName("btn-record")
        self._btn_record.setEnabled(False)
        self._btn_record.clicked.connect(self._on_record_toggle)
        rec_layout.addWidget(self._btn_record)

        self._btn_save = QPushButton("💾  Save Data")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        rec_layout.addWidget(self._btn_save)

        self._save_status = QLabel("")
        self._save_status.setStyleSheet("color: #88ff88; font-size: 11px;")
        self._save_status.setWordWrap(True)
        rec_layout.addWidget(self._save_status)

        layout.addWidget(rec_group)

        # ── Settings group ────────────────────────────────────────────────────
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)

        notch_row = QHBoxLayout()
        notch_row.addWidget(QLabel("Notch filter:"))
        self._notch_combo = QComboBox()
        self._notch_combo.addItems(["50 Hz", "60 Hz"])
        self._notch_combo.currentIndexChanged.connect(self._on_notch_change)
        notch_row.addWidget(self._notch_combo)
        settings_layout.addLayout(notch_row)

        layout.addWidget(settings_group)

        layout.addStretch()
        return sidebar

    # ═════════════════════════════════════════════════════════════════════════
    # Timers
    # ═════════════════════════════════════════════════════════════════════════

    def _build_timers(self):
        self._timer_fast = QTimer(self)
        self._timer_fast.setInterval(100)
        self._timer_fast.timeout.connect(self._update_fast)

        self._timer_slow = QTimer(self)
        self._timer_slow.setInterval(2000)
        self._timer_slow.timeout.connect(self._update_slow)

        self._timer_rec = QTimer(self)
        self._timer_rec.setInterval(1000)
        self._timer_rec.timeout.connect(self._update_rec_timer)

    # ═════════════════════════════════════════════════════════════════════════
    # Fast update (100 ms — EEG sweep)
    # ═════════════════════════════════════════════════════════════════════════

    def _update_fast(self):
        try:
            self._do_update_fast()
        except Exception:
            traceback.print_exc()

    def _do_update_fast(self):
        new_raw, _ = self._reader.get_new_samples(self._last_seen)
        n_new = new_raw.shape[0]
        self._last_seen += n_new

        if n_new > 0:
            filtered, self._filter_states = self._processor.filter_new(
                new_raw, self._filter_states
            )
            end = self._write_idx + n_new
            if end <= SWEEP_SIZE:
                self._y_sweep[self._write_idx:end] = filtered
            else:
                first = SWEEP_SIZE - self._write_idx
                self._y_sweep[self._write_idx:] = filtered[:first]
                self._y_sweep[:end - SWEEP_SIZE] = filtered[first:]
            self._write_idx = end % SWEEP_SIZE

            # Erase cursor gap
            eraser_end = min(self._write_idx + ERASER_WIDTH, SWEEP_SIZE)
            self._y_sweep[self._write_idx:eraser_end] = np.nan
            wrap = (self._write_idx + ERASER_WIDTH) - SWEEP_SIZE
            if wrap > 0:
                self._y_sweep[:wrap] = np.nan

        # Signal quality
        valid = ~np.isnan(self._y_sweep[:, :4])
        if valid.any():
            safe = np.where(valid, self._y_sweep[:, :4], 0.0)
            quality = self._processor.signal_quality(safe)
        else:
            quality = ["poor"] * 4

        self._eeg_panel.refresh(self._y_sweep, quality)

    # ═════════════════════════════════════════════════════════════════════════
    # Slow update (2000 ms — bands + mental metrics)
    # ═════════════════════════════════════════════════════════════════════════

    def _update_slow(self):
        epoch_raw, _ = self._reader.get_window(seconds=4)
        if epoch_raw.shape[0] < int(self._reader.sample_rate):
            return

        epoch  = self._processor.filter_signal(epoch_raw)
        powers = self._processor.band_powers(epoch)

        now = time.time()
        for band in BANDS:
            self._band_history[band].append(
                (now, float(np.mean(powers[band][:4])))
            )

        frontal = {
            b: float(np.mean(powers[b][[1, 2]]))
            for b in ("theta", "alpha", "beta")
        }
        metrics = self._processor.compute_metrics_from_powers(frontal, now)

        self._bands_panel.refresh(powers, self._band_history)
        self._mental_panel.refresh(metrics, self._processor)

    # ═════════════════════════════════════════════════════════════════════════
    # Sidebar slots
    # ═════════════════════════════════════════════════════════════════════════

    def _on_scan(self):
        self._btn_scan.setEnabled(False)
        self._scan_status.setText("🔍 Scanning (10 s)…")
        self._device_combo.clear()
        self._reader.scan_devices(on_found=self._on_scan_done)

    def _on_scan_done(self, devices: list):
        self._scan_done.emit(devices)   # cross-thread safe via queued connection

    def _apply_scan_results(self, devices: list):
        self._btn_scan.setEnabled(True)
        if not devices:
            self._scan_status.setText("⚠ No Muse found. Try again.")
            return
        self._device_combo.clear()
        for d in devices:
            self._device_combo.addItem(
                f"{d['name']}  ({d['address']})", userData=d["address"]
            )
        self._scan_status.setText(f"✓ {len(devices)} device(s) found")

    def _on_connect_toggle(self):
        if not self._reader.is_connected:
            address = self._device_combo.currentData()
            self._btn_connect.setEnabled(False)
            self._btn_connect.setText("Connecting…")
            # Connect in background; use a one-shot timer to poll
            self._reader.start(address=address or None)
            if self._reader.is_connected:
                self._on_connected()
            else:
                err = self._reader.last_error or "BLE connection failed"
                self._btn_connect.setEnabled(True)
                self._btn_connect.setText("▶  Connect")
                self._set_status(f"● ERR: {err[:30]}", "#cc3333")
        else:
            self._reader.stop()
            self._on_disconnected()

    def _on_connected(self):
        # Reset sweep state
        self._y_sweep[:] = np.nan
        self._write_idx  = 0
        self._last_seen  = self._reader.sample_count
        self._filter_states = self._processor.make_filter_states(_N_CH)

        name = self._reader.device_name or "Muse"
        self._btn_connect.setText(f"⏹  Disconnect ({name})")
        self._btn_connect.setProperty("active", "true")
        self._btn_connect.setEnabled(True)
        self._btn_connect.style().unpolish(self._btn_connect)
        self._btn_connect.style().polish(self._btn_connect)
        self._set_status("● LIVE", "#22cc66")
        self._btn_record.setEnabled(True)

        self._timer_fast.start()
        self._timer_slow.start()

    def _on_disconnected(self):
        self._timer_fast.stop()
        self._timer_slow.stop()
        self._timer_rec.stop()
        self._is_recording = False
        self._btn_record.setText("⏺  Start Recording")
        self._btn_record.setProperty("recording", "false")
        self._btn_record.setEnabled(False)
        self._btn_connect.setText("▶  Connect")
        self._btn_connect.setProperty("active", "false")
        self._btn_connect.setEnabled(True)
        self._btn_connect.style().unpolish(self._btn_connect)
        self._btn_connect.style().polish(self._btn_connect)
        self._set_status("● OFFLINE", QUALITY_COLORS["secondary"])
        self._eeg_panel.reset()

    def _on_record_toggle(self):
        if not self._is_recording:
            self._is_recording = True
            self._record_start = time.time()
            self._btn_record.setText("⏹  Stop Recording")
            self._btn_record.setProperty("recording", "true")
            self._btn_record.style().unpolish(self._btn_record)
            self._btn_record.style().polish(self._btn_record)
            self._btn_save.setEnabled(False)
            self._timer_rec.start()
        else:
            self._is_recording = False
            self._timer_rec.stop()
            self._btn_record.setText("⏺  Start Recording")
            self._btn_record.setProperty("recording", "false")
            self._btn_record.style().unpolish(self._btn_record)
            self._btn_record.style().polish(self._btn_record)
            self._btn_save.setEnabled(True)

    def _on_save(self):
        from src.data_manager import save_recording
        data, ts = self._reader.get_all()
        if data.shape[0] == 0:
            self._save_status.setText("⚠ No data to save.")
            return
        metric_h = {
            "workload": list(self._processor.metric_history["workload"]),
            "fatigue":  list(self._processor.metric_history["fatigue"]),
        }
        save_recording(ts, data, sample_rate=self._reader.sample_rate,
                       metric_history=metric_h)
        self._save_status.setText(f"✓ Saved ({data.shape[0]} samples)")

    def _on_notch_change(self, index: int):
        freq = 50.0 if index == 0 else 60.0
        from src.processing import _notch_sos
        self._processor._notch_sos = _notch_sos(freq, self._processor.fs)
        self._filter_states = self._processor.make_filter_states(_N_CH)

    def _update_rec_timer(self):
        if self._is_recording:
            elapsed = int(time.time() - self._record_start)
            h, rem = divmod(elapsed, 3600)
            m, s   = divmod(rem, 60)
            self._timer_lbl.setText(f"{h:02d}:{m:02d}:{s:02d}")

    def _set_status(self, text: str, color: str):
        self._conn_badge.setText(text)
        self._conn_badge.setStyleSheet(
            f"background-color: {color}; "
            "color: white; border-radius: 4px; padding: 2px 6px; font-weight: bold;"
        )

    # ── Clean shutdown ────────────────────────────────────────────────────────

    def closeEvent(self, event):
        self._timer_fast.stop()
        self._timer_slow.stop()
        self._timer_rec.stop()
        if self._reader.is_connected:
            self._reader.stop()
        event.accept()
