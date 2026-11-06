"""
main_window.py
QMainWindow for the Muse S EEG Desktop Dashboard.
"""

import threading
import time
import traceback
from collections import deque
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGridLayout, QGroupBox,
    QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QProgressBar, QPushButton, QScrollArea, QSplitter,
    QTabWidget, QVBoxLayout, QWidget,
)

from src.gui.bands_panel import BandsPanel
from src.gui.eeg_panel import EEGPanel
from src.gui.mental_panel import MentalPanel
from src.gui.ppg_panel import PPGPanel, compute_hr_hrv
from src.gui.theme import (
    CH_NAMES, ERASER_WIDTH, QUALITY_COLORS, SWEEP_SIZE, SWEEP_WINDOW_S,
)
from src.processing import BANDS, EEGProcessor
from src.stream_reader import StreamReader

_N_CH = 5


class MainWindow(QMainWindow):

    _scan_done      = pyqtSignal(list)         # scan results (final list)
    _scan_live      = pyqtSignal(list)         # live discoveries during scan
    _connect_result = pyqtSignal(bool, str)    # (success, error_msg) from BLE thread

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IsharatMukh — EEG Dashboard")
        self.resize(1400, 820)

        # ── Backend singletons ────────────────────────────────────────────────
        self._ble_reader = StreamReader()
        self._lsl_reader = None                # created lazily (LSL source)
        self._reader     = self._ble_reader    # active reader
        self._lsl_pub    = None                # LSLOutlet when enabled
        self._lsl_push_seen = 0                # cursor for LSL-outlet pushing
        self._processor  = EEGProcessor(notch_freq=50.0)

        # ── Sweep state ───────────────────────────────────────────────────────
        self._y_sweep       = np.full((SWEEP_SIZE, _N_CH), np.nan)
        self._write_idx     = 0
        self._last_seen     = 0
        self._filter_states = self._processor.make_filter_states(_N_CH)
        self._eeg_mode_prev = 'scroll'   # detect mode switch for state reset

        # ── Band history ──────────────────────────────────────────────────────
        self._band_history: dict[str, deque] = {b: deque(maxlen=3600) for b in BANDS}

        # ── Recording state ───────────────────────────────────────────────────
        self._is_recording  = False
        self._record_start  = 0.0

        # ── Markers / triggers ────────────────────────────────────────────────
        self._marker_defs:      list[str] = ["Start", "Rest", "Task", "End"]
        self._marker_events:    list[tuple[float, str]] = []
        self._marker_btns:      list[QPushButton] = []
        self._marker_shortcuts: list[QShortcut] = []

        # ── PPG state ─────────────────────────────────────────────────────────
        self._ppg_last_seen = 0
        self._stream_start  = 0.0    # used for relative time in HR/HRV history

        # ── Save settings ─────────────────────────────────────────────────────
        self._save_settings = {
            "path":    str(Path(__file__).parent.parent.parent / "data"),
            "formats": {"csv", "edf"},
            "include": {
                "eeg":            True,
                "eeg_ch":         [True, True, True, True],  # TP9/AF7/AF8/TP10
                "filtered":       True,
                "mental":         True,
                "markers":        True,
                "ppg":            True,
                "threshold_flag": True,
            },
        }

        self._scan_done.connect(self._apply_scan_results)
        self._scan_live.connect(self._on_scan_live)
        self._connect_result.connect(self._on_connect_result)
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
        _sb_scroll = QScrollArea()
        _sb_scroll.setWidget(self._build_sidebar())
        _sb_scroll.setWidgetResizable(True)
        _sb_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        _sb_scroll.setMinimumWidth(200)
        _sb_scroll.setMaximumWidth(360)
        splitter.addWidget(_sb_scroll)

        # Main tab area
        self._tabs = QTabWidget()
        self._eeg_panel    = EEGPanel()
        self._bands_panel  = BandsPanel()
        self._mental_panel = MentalPanel()
        self._ppg_panel    = PPGPanel()
        self._tabs.addTab(self._eeg_panel,    "📈  Raw EEG")
        self._tabs.addTab(self._bands_panel,  "🌊  EEG Bands")
        self._tabs.addTab(self._mental_panel, "🧠  Mental State")
        self._tabs.addTab(self._ppg_panel,    "💓  PPG")
        splitter.addWidget(self._tabs)

        splitter.setSizes([220, 1180])
        splitter.setCollapsible(0, False)
        root_h.addWidget(splitter)

    def _build_sidebar(self) -> QWidget:
        sidebar = QWidget()
        sidebar.setMinimumWidth(160)
        sidebar.setMaximumWidth(340)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        # Title
        title = QLabel("IsharatMukh")
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

        src_row = QHBoxLayout()
        src_row.addWidget(QLabel("Source:"))
        self._source_combo = QComboBox()
        self._source_combo.addItems(["Direct BLE (Muse)", "LSL Stream"])
        self._source_combo.currentIndexChanged.connect(self._on_source_change)
        src_row.addWidget(self._source_combo)
        dev_layout.addLayout(src_row)

        self._btn_scan = QPushButton("🔍 Scan for Muse")
        self._btn_scan.clicked.connect(self._on_scan)
        dev_layout.addWidget(self._btn_scan)

        self._scan_bar = QProgressBar()
        self._scan_bar.setRange(0, 0)          # indeterminate/busy indicator
        self._scan_bar.setFixedHeight(8)
        self._scan_bar.setTextVisible(False)
        self._scan_bar.setVisible(False)
        dev_layout.addWidget(self._scan_bar)

        self._scan_status = QLabel("")
        self._scan_status.setStyleSheet("color: #88aaff; font-size: 11px;")
        self._scan_status.setWordWrap(True)
        dev_layout.addWidget(self._scan_status)

        self._device_list = QListWidget()
        self._device_list.setMaximumHeight(150)
        dev_layout.addWidget(self._device_list)

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

        save_row = QHBoxLayout()
        self._btn_save = QPushButton("💾  Save Data")
        self._btn_save.setEnabled(False)
        self._btn_save.clicked.connect(self._on_save)
        save_row.addWidget(self._btn_save, stretch=1)
        btn_save_cfg = QPushButton("⚙")
        btn_save_cfg.setFixedWidth(30)
        btn_save_cfg.setToolTip("Configure save path, format and what to include")
        btn_save_cfg.clicked.connect(self._on_save_settings)
        save_row.addWidget(btn_save_cfg)
        rec_layout.addLayout(save_row)

        self._btn_clear = QPushButton("🗑️  Clear Data")
        self._btn_clear.setEnabled(False)
        self._btn_clear.setToolTip("Delete all buffered EEG/PPG data and restart sweep")
        self._btn_clear.clicked.connect(self._on_clear_data)
        rec_layout.addWidget(self._btn_clear)

        self._cb_lsl_out = QCheckBox("Publish to LSL (EEG + markers)")
        self._cb_lsl_out.setToolTip(
            "Re-broadcast as LSL streams (MuseDashboard-EEG / -Markers)\n"
            "so LabRecorder and other tools can subscribe."
        )
        rec_layout.addWidget(self._cb_lsl_out)

        self._save_status = QLabel("")
        self._save_status.setStyleSheet("color: #88ff88; font-size: 11px;")
        self._save_status.setWordWrap(True)
        rec_layout.addWidget(self._save_status)

        layout.addWidget(rec_group)

        # ── Markers / triggers group ──────────────────────────────────────────
        marker_group = QGroupBox("Markers / Triggers")
        marker_layout = QVBoxLayout(marker_group)
        marker_layout.setSpacing(6)

        self._marker_grid = QGridLayout()
        self._marker_grid.setSpacing(4)
        marker_layout.addLayout(self._marker_grid)

        mk_row = QHBoxLayout()
        btn_add_mk = QPushButton("+ Add")
        btn_add_mk.setFixedHeight(24)
        btn_add_mk.setToolTip("Define a new marker button")
        btn_add_mk.clicked.connect(self._on_add_marker)
        mk_row.addWidget(btn_add_mk)
        btn_del_mk = QPushButton("− Remove")
        btn_del_mk.setFixedHeight(24)
        btn_del_mk.setToolTip("Remove a marker button")
        btn_del_mk.clicked.connect(self._on_remove_marker)
        mk_row.addWidget(btn_del_mk)
        marker_layout.addLayout(mk_row)

        self._marker_status = QLabel("Markers saved with session data\n(also keys 1–9)")
        self._marker_status.setStyleSheet("color: #ffcc66; font-size: 11px;")
        self._marker_status.setWordWrap(True)
        marker_layout.addWidget(self._marker_status)

        layout.addWidget(marker_group)
        self._rebuild_marker_buttons()

        # ── Settings group ────────────────────────────────────────────────────
        settings_group = QGroupBox("Filters")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)

        self._flt_enable = QCheckBox("Enable real-time filters")
        self._flt_enable.setChecked(True)
        self._flt_enable.toggled.connect(self._apply_filter_settings)
        settings_layout.addWidget(self._flt_enable)

        bp_row = QHBoxLayout()
        bp_row.addWidget(QLabel("Bandpass:"))
        self._flt_low = QDoubleSpinBox()
        self._flt_low.setRange(0.1, 100.0)
        self._flt_low.setSingleStep(0.5)
        self._flt_low.setDecimals(1)
        self._flt_low.setValue(0.5)
        self._flt_low.setSuffix(" Hz")
        self._flt_low.valueChanged.connect(self._apply_filter_settings)
        bp_row.addWidget(self._flt_low)
        self._flt_high = QDoubleSpinBox()
        self._flt_high.setRange(1.0, 120.0)
        self._flt_high.setSingleStep(1.0)
        self._flt_high.setDecimals(1)
        self._flt_high.setValue(45.0)
        self._flt_high.setSuffix(" Hz")
        self._flt_high.valueChanged.connect(self._apply_filter_settings)
        bp_row.addWidget(self._flt_high)
        settings_layout.addLayout(bp_row)

        notch_row = QHBoxLayout()
        notch_row.addWidget(QLabel("Notch filter:"))
        self._notch_combo = QComboBox()
        self._notch_combo.addItems(["50 Hz", "60 Hz"])
        self._notch_combo.currentIndexChanged.connect(self._apply_filter_settings)
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

        self._timer_ppg = QTimer(self)
        self._timer_ppg.setInterval(100)
        self._timer_ppg.timeout.connect(self._update_ppg)

        self._timer_ppg_chk = QTimer(self)
        self._timer_ppg_chk.setSingleShot(True)
        self._timer_ppg_chk.setInterval(10_000)   # 10 s after connect
        self._timer_ppg_chk.timeout.connect(self._check_ppg_available)

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
        # Re-broadcast new raw samples to the LSL outlet, if enabled
        if self._lsl_pub is not None:
            chunk, _ = self._reader.get_new_samples(self._lsl_push_seen)
            self._lsl_push_seen += chunk.shape[0]
            if chunk.shape[0]:
                self._lsl_pub.push_eeg(chunk)

        mode = self._eeg_panel.display_mode

        # Reset state whenever the user switches modes
        if mode != self._eeg_mode_prev:
            self._eeg_mode_prev  = mode
            self._y_sweep[:]     = np.nan
            self._write_idx      = 0
            self._last_seen      = self._reader.sample_count
            self._filter_states  = self._processor.make_filter_states(_N_CH)

        if mode == 'sweep':
            # ── Draw-and-erase / ECG-style circular buffer ────────────────────
            new_raw, _ = self._reader.get_new_samples(self._last_seen)
            n_new = new_raw.shape[0]
            self._last_seen += n_new

            if n_new > 0:
                filtered, self._filter_states = self._processor.filter_new(
                    new_raw, self._filter_states
                )
                if n_new > SWEEP_SIZE:          # lag overflow — keep latest
                    filtered = filtered[-SWEEP_SIZE:]
                    n_new = SWEEP_SIZE
                end = self._write_idx + n_new
                if end <= SWEEP_SIZE:
                    self._y_sweep[self._write_idx:end] = filtered
                else:
                    first = SWEEP_SIZE - self._write_idx
                    self._y_sweep[self._write_idx:] = filtered[:first]
                    self._y_sweep[:end - SWEEP_SIZE] = filtered[first:]
                self._write_idx = end % SWEEP_SIZE

                eraser_end = min(self._write_idx + ERASER_WIDTH, SWEEP_SIZE)
                self._y_sweep[self._write_idx:eraser_end] = np.nan
                wrap = (self._write_idx + ERASER_WIDTH) - SWEEP_SIZE
                if wrap > 0:
                    self._y_sweep[:wrap] = np.nan

            valid = ~np.isnan(self._y_sweep[:, :4])
            stats = (
                self._processor.channel_stats(
                    np.where(valid, self._y_sweep[:, :4], 0.0)
                )
                if valid.any()
                else self._processor.channel_stats(np.empty((0, 4)))
            )

        else:
            # ── Scrolling strip chart: latest window right-aligned ───────────────
            raw_win, _ = self._reader.get_window(seconds=SWEEP_WINDOW_S)
            n = raw_win.shape[0]
            self._y_sweep[:] = np.nan

            if n >= 10:
                filtered = self._processor.filter_signal(raw_win)
                n_disp = min(n, SWEEP_SIZE)
                self._y_sweep[SWEEP_SIZE - n_disp:] = filtered[-n_disp:]
                stats = self._processor.channel_stats(
                    filtered[-min(int(self._reader.sample_rate), n):, :4]
                )
            else:
                stats = self._processor.channel_stats(np.empty((0, 4)))

        self._eeg_panel.refresh(self._y_sweep, stats)

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
        metrics = self._processor.compute_metrics_from_powers(frontal, now,
                                                               ch_powers=powers)

        self._bands_panel.refresh(powers, self._band_history)
        self._mental_panel.refresh(metrics, self._processor)
        self._update_ppg_metrics()

    # ═════════════════════════════════════════════════════════════════════════
    # Sidebar slots
    # ═════════════════════════════════════════════════════════════════════════

    def _on_source_change(self, index: int):
        self._btn_scan.setText(
            "🔍 Scan for Muse" if index == 0 else "🔍 Resolve LSL streams"
        )
        self._device_list.clear()
        self._scan_status.setText("")

    def _on_scan(self):
        self._btn_scan.setEnabled(False)
        self._device_list.clear()
        self._scan_bar.setVisible(True)
        if self._source_combo.currentIndex() == 1:
            self._scan_status.setText("🔍 Resolving LSL streams…")
            self._scan_lsl()
        else:
            self._scan_status.setText("🔍 Scanning for Muse…")
            self._ble_reader.scan_devices(
                on_found=self._on_scan_done,
                on_progress=lambda devs: self._scan_live.emit(devs),
            )

    def _on_scan_done(self, devices: list):
        self._scan_done.emit(devices)   # cross-thread safe via queued connection

    def _on_scan_live(self, devices: list):
        self._populate_device_list(devices)
        self._scan_status.setText(f"🔍 Scanning… {len(devices)} found")

    def _apply_scan_results(self, devices: list):
        self._btn_scan.setEnabled(True)
        self._scan_bar.setVisible(False)
        if not devices:
            self._scan_status.setText("⚠ Nothing found. Try again.")
            return
        self._populate_device_list(devices)
        self._scan_status.setText(f"✓ {len(devices)} found — select & connect")

    def _populate_device_list(self, devices: list):
        prev_addr = self._current_device_address()
        self._device_list.clear()
        for d in devices:
            item = QListWidgetItem(self._device_item_text(d))
            item.setData(Qt.ItemDataRole.UserRole, d)
            self._device_list.addItem(item)
            if prev_addr and d.get("address") == prev_addr:
                self._device_list.setCurrentItem(item)
        if self._device_list.count() and not self._device_list.currentItem():
            self._device_list.setCurrentRow(0)

    def _scan_lsl(self):
        """Resolve available LSL streams in a background thread."""
        def _resolve():
            try:
                from src.lsl_io import LSLReader
                streams = LSLReader.resolve(timeout=3.0)
            except Exception as exc:
                print(f"[LSL] resolve error: {exc}")
                streams = []
            self._scan_done.emit(streams)
        threading.Thread(target=_resolve, daemon=True, name="lsl-scan").start()

    def _connect_lsl(self):
        """Connect to the LSL stream selected in the device list."""
        item = self._device_list.currentItem()
        dev  = item.data(Qt.ItemDataRole.UserRole) if item else None
        info = dev.get("_stream_info") if isinstance(dev, dict) else None
        from src.lsl_io import LSLReader
        if self._lsl_reader is None:
            self._lsl_reader = LSLReader()
        self._reader = self._lsl_reader
        self._lsl_reader.start_async(
            stream_info=info,
            on_result=lambda ok, err: self._connect_result.emit(ok, err),
        )

    @staticmethod
    def _device_item_text(d: dict) -> str:
        name, addr = d.get("name") or "?", d.get("address", "")
        rssi = d.get("rssi")
        extra = (f"  {d['n_ch']}ch @ {d['fs']:g} Hz"
                 if d.get("n_ch") else "")
        if rssi is None:
            return f"{name}  ({addr}){extra}"
        level = 3 if rssi > -55 else 2 if rssi > -70 else 1 if rssi > -85 else 0
        bars  = "█" * (level + 1) + "░" * (3 - level)
        return f"{name}  ({addr})\n{rssi} dBm {bars}"

    def _current_device_address(self) -> str | None:
        item = self._device_list.currentItem()
        if not item:
            return None
        d = item.data(Qt.ItemDataRole.UserRole)
        return d.get("address") if isinstance(d, dict) else None

    def _on_connect_toggle(self):
        if not self._reader.is_connected:
            self._btn_connect.setEnabled(False)
            self._btn_connect.setText("Connecting…")
            self._set_status("● Connecting…", "#f1a24e")
            if self._source_combo.currentIndex() == 1:
                self._connect_lsl()
                return
            self._reader = self._ble_reader
            # Non-blocking: _on_connect_result is called from background thread
            # which emits _connect_result signal (cross-thread safe)
            self._reader.start_async(
                address=self._current_device_address() or None,
                on_result=lambda ok, err: self._connect_result.emit(ok, err),
            )
        else:
            self._reader.stop()
            self._on_disconnected()

    def _on_connect_result(self, success: bool, error: str):
        """Slot called on main thread when BLE connect attempt finishes."""
        if success:
            self._on_connected()
        else:
            err = error or "BLE connection failed"
            self._btn_connect.setEnabled(True)
            self._btn_connect.setText("▶  Connect")
            self._set_status(f"● ERR: {err[:30]}", "#cc3333")

    def _on_connected(self):
        # Reset sweep state
        self._y_sweep[:] = np.nan
        self._write_idx  = 0
        self._last_seen  = self._reader.sample_count
        self._ppg_last_seen = self._reader.ppg_count
        self._stream_start  = time.time()
        self._filter_states = self._processor.make_filter_states(_N_CH)
        self._processor.reset_baseline()

        self._lsl_pub = None
        self._lsl_push_seen = self._reader.sample_count
        if self._cb_lsl_out.isChecked():
            try:
                from src.lsl_io import LSLOutlet
                self._lsl_pub = LSLOutlet(fs=self._reader.sample_rate,
                                          n_ch=_N_CH)
                self._scan_status.setText("📡 LSL outlet active")
            except Exception as exc:
                self._scan_status.setText(f"⚠ LSL outlet failed: {exc}")

        name = self._reader.device_name or "Muse"
        self._btn_connect.setText(f"⏹  Disconnect ({name})")
        self._btn_connect.setProperty("active", "true")
        self._btn_connect.setEnabled(True)
        self._btn_connect.style().unpolish(self._btn_connect)
        self._btn_connect.style().polish(self._btn_connect)
        self._set_status("● LIVE", "#22cc66")
        self._btn_record.setEnabled(True)
        self._btn_clear.setEnabled(True)

        self._timer_fast.start()
        self._timer_slow.start()
        self._timer_ppg.start()
        self._timer_ppg_chk.start()

    def _on_disconnected(self):
        self._timer_fast.stop()
        self._timer_slow.stop()
        self._timer_ppg.stop()
        self._timer_ppg_chk.stop()
        self._timer_rec.stop()
        # Re-enable PPG tab in case it was disabled by the availability check
        ppg_idx = self._tabs.indexOf(self._ppg_panel)
        self._tabs.setTabEnabled(ppg_idx, True)
        self._tabs.setTabText(ppg_idx, "💓  PPG")
        self._is_recording = False
        self._btn_record.setText("⏺  Start Recording")
        self._btn_record.setProperty("recording", "false")
        self._btn_record.setEnabled(False)
        self._btn_clear.setEnabled(False)
        self._btn_connect.setText("▶  Connect")
        self._btn_connect.setProperty("active", "false")
        self._btn_connect.setEnabled(True)
        self._btn_connect.style().unpolish(self._btn_connect)
        self._btn_connect.style().polish(self._btn_connect)
        self._set_status("● OFFLINE", QUALITY_COLORS["secondary"])
        self._lsl_pub = None
        self._eeg_panel.reset()
        self._ppg_panel.reset()

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

    def _on_save_settings(self):
        from src.gui.save_dialog import SaveSettingsDialog
        dlg = SaveSettingsDialog(self._save_settings, parent=self)
        if dlg.exec():
            self._save_settings = dlg.settings

    def _check_ppg_available(self):
        if self._reader.ppg_count == 0:
            idx = self._tabs.indexOf(self._ppg_panel)
            self._tabs.setTabEnabled(idx, False)
            self._tabs.setTabText(idx, "💓  PPG (n/a)")

    def _on_save(self):
        from src.data_manager import save_recording
        s   = self._save_settings
        inc = s.get("include", {})

        data, ts         = self._reader.get_all()
        ppg_data, ppg_ts = self._reader.get_all_ppg()

        if data.shape[0] == 0 and ppg_data.shape[0] == 0:
            self._save_status.setText("⚠ No data to save.")
            return

        # EEG channel selection (first 4 cols = TP9/AF7/AF8/TP10)
        ch_mask = inc.get("eeg_ch", [True]*4)
        sel     = [i for i, ok in enumerate(ch_mask) if ok] or list(range(4))
        data_save  = data[:, sel]
        ch_names   = [CH_NAMES[i] for i in sel]

        filtered_save = (
            self._processor.filter_signal(data_save)
            if inc.get("filtered", True) and self._processor.filters_enabled
            else None
        )
        metric_h = (
            {k: list(v) for k, v in self._processor.metric_history.items()}
            if inc.get("mental", True) else None
        )
        ppg_ok = inc.get("ppg", True) and ppg_data.shape[0] > 0

        meta = {
            "session_source": "BLE" if self._reader is self._ble_reader else "LSL",
            "device_name":     self._reader.device_name,
            "device_address":  self._reader.device_address,
            "sample_rate_hz":  self._reader.sample_rate,
            "channels":        ch_names,
            "n_samples":       int(data_save.shape[0]),
            "filters": {
                "enabled":  self._processor.filters_enabled,
                "low_hz":   self._processor.filter_low,
                "high_hz":  self._processor.filter_high,
                "notch_hz": self._processor.notch_freq,
            },
            "marker_events": [
                {"t": t, "label": l} for t, l in self._marker_events
            ],
            "include": inc,
            "formats": sorted(s.get("formats", [])),
            "note": ("metric columns are sparse (values at ~2 s epochs, NaN "
                     "between). marker column holds trigger labels at the "
                     "nearest sample."),
        }

        saved = save_recording(
            ts, data_save,
            filtered_data=filtered_save,
            sample_rate=self._reader.sample_rate,
            metric_history=metric_h,
            marker_events=(self._marker_events
                           if inc.get("markers", True) else None),
            channel_names=ch_names,
            meta=meta,
            ppg_timestamps=ppg_ts if ppg_ok else None,
            ppg_data=ppg_data      if ppg_ok else None,
            save_path=s.get("path"),
            formats=s.get("formats", {"csv", "edf"}),
            threshold_flag=inc.get("threshold_flag", True),
        )
        n_ppg = ppg_data.shape[0] if ppg_ok else 0
        self._save_status.setText(
            f"✓ Saved — EEG: {data_save.shape[0]}  PPG: {n_ppg}\n"
            f"{saved.get('csv', '')}"
        )

    def _on_clear_data(self):
        """Delete all buffered EEG/PPG data and restart the sweep."""
        self._reader.clear()
        self._processor.reset_baseline()
        self._y_sweep[:] = np.nan
        self._write_idx  = 0
        self._last_seen  = 0
        self._ppg_last_seen = 0
        self._stream_start  = time.time()
        self._filter_states = self._processor.make_filter_states(_N_CH)
        self._band_history  = {b: deque(maxlen=3600) for b in BANDS}
        self._marker_events.clear()
        self._marker_status.setText("Markers saved with session data\n(also keys 1–9)")
        self._eeg_panel.reset()
        self._ppg_panel.reset()
        self._save_status.setText("Data cleared.")

    # ── Markers / triggers ────────────────────────────────────────────────────

    def _rebuild_marker_buttons(self):
        """Re-create marker buttons + number-key shortcuts from _marker_defs."""
        while self._marker_grid.count():
            w = self._marker_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        for sc in self._marker_shortcuts:
            sc.setParent(None)
            sc.deleteLater()
        self._marker_shortcuts.clear()
        self._marker_btns.clear()

        for i, label in enumerate(self._marker_defs):
            btn = QPushButton(f"{i + 1}. {label}" if i < 9 else label)
            btn.setFixedHeight(28)
            btn.setToolTip(f"Insert a '{label}' marker at the current time")
            btn.clicked.connect(
                lambda _checked, lbl=label: self._fire_marker(lbl)
            )
            self._marker_grid.addWidget(btn, i // 2, i % 2)
            self._marker_btns.append(btn)
            if i < 9:
                sc = QShortcut(QKeySequence(str(i + 1)), self)
                sc.activated.connect(
                    lambda lbl=label: self._fire_marker(lbl)
                )
                self._marker_shortcuts.append(sc)

    def _on_add_marker(self):
        label, ok = QInputDialog.getText(self, "Add marker", "Marker label:")
        label = label.strip()
        if ok and label and label not in self._marker_defs:
            self._marker_defs.append(label)
            self._rebuild_marker_buttons()

    def _on_remove_marker(self):
        if not self._marker_defs:
            return
        label, ok = QInputDialog.getItem(
            self, "Remove marker", "Select marker to remove:",
            self._marker_defs, 0, False,
        )
        if ok and label in self._marker_defs:
            self._marker_defs.remove(label)
            self._rebuild_marker_buttons()

    def _fire_marker(self, label: str):
        t = self._reader.now()
        self._marker_events.append((t, label))
        self._marker_status.setText(
            f"Last: {label} @ {time.strftime('%H:%M:%S')}"
            f"  (total {len(self._marker_events)})"
        )
        if self._lsl_pub is not None:
            self._lsl_pub.push_marker(label)

    def _apply_filter_settings(self, *_args):
        """Push current filter UI settings into the processor + reset states."""
        try:
            self._processor.configure_filters(
                low_hz=self._flt_low.value(),
                high_hz=self._flt_high.value(),
                notch_hz=50.0 if self._notch_combo.currentIndex() == 0 else 60.0,
                enabled=self._flt_enable.isChecked(),
            )
        except ValueError as exc:
            self._save_status.setText(f"⚠ {exc}")
            return
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

    # ── PPG fast update ───────────────────────────────────────────────────────

    def _update_ppg(self):
        try:
            new_ppg, _ = self._reader.get_new_ppg_samples(self._ppg_last_seen)
            self._ppg_last_seen += new_ppg.shape[0]
            ppg_win, _ = self._reader.get_ppg_window(seconds=10)
            self._ppg_panel.refresh_sweep(ppg_win)
        except Exception:
            traceback.print_exc()

    def _update_ppg_metrics(self):
        """Called from _update_slow to compute HR/HRV from PPG window."""
        try:
            ppg_win, _ = self._reader.get_ppg_window(seconds=30)
            if ppg_win.shape[0] < 64 * 5:
                return
            hr, hrv = compute_hr_hrv(ppg_win[:, 1], fs=self._reader.ppg_sample_rate)
            t_rel = time.time() - self._stream_start
            self._ppg_panel.refresh_metrics(hr, hrv, t_rel)
        except Exception:
            traceback.print_exc()

    def closeEvent(self, event):
        self._timer_fast.stop()
        self._timer_slow.stop()
        self._timer_ppg.stop()
        self._timer_rec.stop()
        self._lsl_pub = None
        if self._ble_reader.is_connected:
            self._ble_reader.stop()
        if self._lsl_reader is not None and self._lsl_reader.is_connected:
            self._lsl_reader.stop()
        event.accept()
