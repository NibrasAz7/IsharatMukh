"""
main_window.py
QMainWindow for the Muse S EEG Desktop Dashboard.
"""

import threading
import time
import traceback
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QApplication, QCheckBox, QComboBox, QFileDialog, QGridLayout,
    QGroupBox, QHBoxLayout, QLabel, QListWidget, QListWidgetItem,
    QMainWindow, QMessageBox, QProgressBar, QPushButton, QScrollArea,
    QSplitter, QTabWidget, QVBoxLayout, QWidget,
)

from src.gui.bands_panel import BandsPanel
from src.gui.eeg_panel import EEGPanel
from src.gui.mental_panel import MentalPanel
from src.gui.ppg_panel import PPGPanel, compute_hr_hrv
from src.gui.theme import (
    CH_NAMES, ERASER_WIDTH, QUALITY_COLORS, SWEEP_SIZE, SWEEP_WINDOW_S,
    build_stylesheet,
)
from src.processing import BANDS, EEGProcessor
from src.recorder import SessionRecorder
from src.settings import banks_from_settings, load_settings, save_settings
from src.stream_reader import StreamReader

_N_CH = 5
APP_VERSION = "0.2.0-alpha"


class MainWindow(QMainWindow):

    _scan_done      = pyqtSignal(list)         # scan results (final list)
    _scan_live      = pyqtSignal(list)         # live discoveries during scan
    _connect_result = pyqtSignal(bool, str)    # (success, error_msg) from BLE thread
    _bt_probe       = pyqtSignal(str)          # bluetooth availability probe

    def __init__(self):
        super().__init__()
        self.setWindowTitle("IsharatMukh — EEG Dashboard")
        self.resize(1400, 820)

        # ── Persistent settings ──────────────────────────────────────────────
        self._settings  = load_settings()
        self._ui_scale  = float(self._settings.get("ui_scale", 1.0))
        self._thr       = dict(self._settings.get("metric_thresholds", {}))

        # ── Backend singletons ────────────────────────────────────────────────
        self._ble_reader = StreamReader()
        self._lsl_reader = None                # created lazily (LSL source)
        self._reader     = self._ble_reader    # active reader
        self._lsl_pub    = None                # LSLOutlet when enabled
        self._lsl_push_seen = 0                # cursor for LSL-outlet pushing
        self._processor  = EEGProcessor(notch_freq=50.0)
        _banks, _active = banks_from_settings(self._settings)
        if _banks:
            try:
                self._processor.load_banks(_banks, _active)
            except Exception:
                pass

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
        self._session       = None          # finalized SessionRecorder result
        self._recorder      = SessionRecorder()
        self._rec_eeg_seen  = 0             # sample_count cursor while recording
        self._rec_ppg_seen  = 0
        self._bt_state      = None          # cached bluetooth probe result

        # ── Markers / triggers ────────────────────────────────────────────────
        self._marker_defs:      list[str] = list(
            self._settings.get("marker_defs", ["Start", "Rest", "Task", "End"]))
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
        self._bt_probe.connect(self._on_bt_probe)
        self._build_ui()
        self._build_timers()
        self._mental_panel.set_thresholds(self._thr)
        if self._ui_scale != 1.0:
            self._apply_ui_scale(self._ui_scale)

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
        self._splitter = splitter
        _sb_scroll = QScrollArea()
        _sb_scroll.setWidget(self._build_sidebar())
        _sb_scroll.setWidgetResizable(True)
        _sb_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        _sb_scroll.setMinimumWidth(int(200 * self._ui_scale))
        _sb_scroll.setMaximumWidth(int(380 * self._ui_scale))
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

        splitter.setSizes([int(230 * self._ui_scale),
                           int(1400 * self._ui_scale)])
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
        btn_app_cfg = QPushButton("🛠")
        btn_app_cfg.setFixedWidth(30)
        btn_app_cfg.setToolTip(
            "App settings — UI scale, autosave, thresholds, filter banks")
        btn_app_cfg.clicked.connect(self._on_app_settings)
        save_row.addWidget(btn_app_cfg)
        rec_layout.addLayout(save_row)

        self._btn_epochs = QPushButton("⇩  Extract epochs…")
        self._btn_epochs.setToolTip(
            "Re-read a saved session_X_data.csv and write its "
            "above-threshold epoch table")
        self._btn_epochs.clicked.connect(self._on_extract_epochs)
        rec_layout.addWidget(self._btn_epochs)

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
        btn_edit_mk = QPushButton("✎  Edit markers…")
        btn_edit_mk.setToolTip(
            "Rename/reorder marker buttons and edit fired events")
        btn_edit_mk.clicked.connect(self._on_edit_markers)
        mk_row.addWidget(btn_edit_mk)
        marker_layout.addLayout(mk_row)

        self._marker_status = QLabel("Markers saved with session data\n(also keys 1–9)")
        self._marker_status.setStyleSheet("color: #ffcc66; font-size: 11px;")
        self._marker_status.setWordWrap(True)
        marker_layout.addWidget(self._marker_status)

        layout.addWidget(marker_group)
        self._rebuild_marker_buttons()

        # ── Filter banks group ────────────────────────────────────────────────
        settings_group = QGroupBox("Filters")
        settings_layout = QVBoxLayout(settings_group)
        settings_layout.setSpacing(6)

        bank_row = QHBoxLayout()
        bank_row.addWidget(QLabel("Active bank:"))
        self._bank_combo = QComboBox()
        self._bank_combo.currentTextChanged.connect(self._on_bank_changed)
        bank_row.addWidget(self._bank_combo, stretch=1)
        btn_banks = QPushButton("Manage…")
        btn_banks.setToolTip("Add/edit/reorder filter banks and stages")
        btn_banks.clicked.connect(self._on_manage_banks)
        bank_row.addWidget(btn_banks)
        settings_layout.addLayout(bank_row)

        self._bank_summary = QLabel("")
        self._bank_summary.setStyleSheet("color: #88aaff; font-size: 11px;")
        self._bank_summary.setWordWrap(True)
        settings_layout.addWidget(self._bank_summary)
        self._refresh_bank_ui()

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
        # Accumulate session data independently of the 60 s display buffer
        if self._is_recording:
            chunk, cts = self._reader.get_new_samples(self._rec_eeg_seen)
            self._rec_eeg_seen += chunk.shape[0]
            self._recorder.append_eeg(chunk, cts)

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
            if self._source_combo.currentIndex() == 0:
                self._check_bluetooth()

    def _check_bluetooth(self):
        """Probe once whether a Bluetooth adapter/service exists; warn if off."""
        if self._bt_state is not None:
            return
        def _probe():
            try:
                import subprocess
                r = subprocess.run(
                    ["powershell", "-NoProfile", "-Command",
                     "$d = Get-PnpDevice -Class Bluetooth -ErrorAction "
                     "SilentlyContinue; $s = (Get-Service bthserv "
                     "-ErrorAction SilentlyContinue).Status; "
                     "if ($d -and $s -eq 'Running') {'OK'} else {'OFF'}"],
                    capture_output=True, text=True, timeout=10)
                self._bt_probe.emit(r.stdout.strip() or "UNKNOWN")
            except Exception:
                self._bt_probe.emit("UNKNOWN")
        threading.Thread(target=_probe, daemon=True,
                         name="bt-probe").start()

    def _on_bt_probe(self, state: str):
        self._bt_state = state
        if state == "OFF":
            QMessageBox.warning(
                self, "Bluetooth unavailable",
                "Bluetooth appears to be turned off or has no adapter.\n\n"
                "Enable it in Windows Settings → Bluetooth (and make sure the "
                "PC's Bluetooth radio is on), then try connecting again.")

    def _on_connected(self):
        # New connection → previous session result no longer applies
        self._session = None
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
        if self._is_recording:
            try:
                self._recorder.stop_recording()
                self._session = self._recorder.result()
                self._auto_save()
            except Exception:
                traceback.print_exc()
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
            self._session = None
            self._recorder.start_recording()
            self._rec_eeg_seen = self._reader.sample_count
            self._rec_ppg_seen = self._reader.ppg_count
            self._btn_record.setText("⏹  Stop Recording")
            self._btn_record.setProperty("recording", "true")
            self._btn_record.style().unpolish(self._btn_record)
            self._btn_record.style().polish(self._btn_record)
            self._btn_save.setEnabled(False)
            self._timer_rec.start()
        else:
            self._is_recording = False
            self._timer_rec.stop()
            self._recorder.stop_recording()
            self._session = self._recorder.result()
            self._btn_record.setText("⏺  Start Recording")
            self._btn_record.setProperty("recording", "false")
            self._btn_record.style().unpolish(self._btn_record)
            self._btn_record.style().polish(self._btn_record)
            self._btn_save.setEnabled(True)
            self._auto_save()

    def _on_save_settings(self):
        from src.gui.save_dialog import SaveSettingsDialog
        dlg = SaveSettingsDialog(self._save_settings, parent=self)
        if dlg.exec():
            self._save_settings = dlg.settings

    def _on_extract_epochs(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select session data CSV", "",
            "Session data (*.csv);;All files (*)")
        if not path or not path.endswith("_data.csv"):
            return
        try:
            from src.epochs import epochs_from_csv
            out = epochs_from_csv(
                path, pad_s=float(self._settings.get("epoch_pad_s", 0.5)),
                thresholds=self._thr)
            self._save_status.setText(f"✓ Epochs → {out.name}")
        except Exception as exc:
            self._save_status.setText(f"⚠ Epoch extraction failed: {exc}")

    def _check_ppg_available(self):
        if self._reader.ppg_count == 0:
            idx = self._tabs.indexOf(self._ppg_panel)
            self._tabs.setTabEnabled(idx, False)
            self._tabs.setTabText(idx, "💓  PPG (n/a)")

    def _on_save(self):
        self._save_data(autosaved=False)

    def _auto_save(self):
        """Autosave the just-finished session if enabled in settings."""
        if not self._settings.get("autosave", True):
            return
        sess = self._session
        if sess is None or sess["eeg_data"].shape[0] == 0:
            return
        try:
            self._save_data(autosaved=True)
        except Exception as exc:
            self._save_status.setText(f"⚠ Auto-save failed: {exc}")

    def _save_data(self, autosaved: bool = False):
        from src.data_manager import save_recording
        s   = self._save_settings
        inc = s.get("include", {})

        # Prefer the session-side recording (full length); fall back to the
        # 60 s display buffer when nothing was recorded this connection.
        sess = self._session
        if sess is not None and sess["eeg_data"].shape[0] > 0:
            data, ts         = sess["eeg_data"], sess["eeg_ts"]
            ppg_data, ppg_ts = sess["ppg_data"], sess["ppg_ts"]
            t_lo = sess["start"] - 1.0
            t_hi = (sess["stop"] or time.time()) + 1.0
            rec_meta = {
                "start":       datetime.fromtimestamp(sess["start"])
                               .isoformat(timespec="seconds"),
                "stop":        datetime.fromtimestamp(sess["stop"] or time.time())
                               .isoformat(timespec="seconds"),
                "duration_s":  round(sess["duration_s"], 1),
                "autosaved":   autosaved,
            }
        else:
            data, ts         = self._reader.get_all()
            ppg_data, ppg_ts = self._reader.get_all_ppg()
            t_lo = t_hi = None
            rec_meta = None

        if data.shape[0] == 0 and ppg_data.shape[0] == 0:
            self._save_status.setText("⚠ No data to save.")
            return

        # EEG channel selection (first 4 cols = TP9/AF7/AF8/TP10)
        ch_mask = inc.get("eeg_ch", [True]*4)
        sel     = [i for i, ok in enumerate(ch_mask) if ok] or list(range(4))
        data_save  = data[:, sel]
        ch_names   = [CH_NAMES[i] for i in sel]

        # One filtered column set per filter bank (banks with ≥1 enabled stage)
        filtered_banks = None
        if inc.get("filtered", True):
            filtered_banks = self._processor.filter_all_banks(data_save)
            if not filtered_banks:
                filtered_banks = None

        metric_h = None
        if inc.get("mental", True):
            metric_h = {
                k: [(t, v) for t, v in self._processor.metric_history[k]
                    if t_lo is None or t_lo <= t <= t_hi]
                for k in self._processor.metric_history
            }
            metric_h = {k: v for k, v in metric_h.items() if v} or None

        # Markers are annotations, not signal — save ALL events fired since
        # the last clear/connect, including pre-roll and post-stop ones.
        marker_events = list(self._marker_events)

        ppg_ok = inc.get("ppg", True) and ppg_data.shape[0] > 0

        meta = {
            "app": {"name": "IsharatMukh", "version": APP_VERSION},
            "session_source": "BLE" if self._reader is self._ble_reader else "LSL",
            "device_name":     self._reader.device_name,
            "device_address":  self._reader.device_address,
            "sample_rate_hz":  self._reader.sample_rate,
            "channels":        ch_names,
            "n_samples":       int(data_save.shape[0]),
            "processing": {
                "filter_banks":       self._processor.bank_to_dicts(),
                "active_bank":        self._processor.active_bank,
                "metric_thresholds":  self._thr,
                "metric_epoch_s":     2.0,
                "baseline_s":         30.0,
            },
            "markers": {
                "defs":   list(self._marker_defs),
                "events": [{"t": t, "label": l} for t, l in marker_events],
            },
            "ui":        {"scale": self._ui_scale},
            "autosave":  autosaved,
            "include":   inc,
            "formats":   sorted(s.get("formats", [])),
            "note": ("metric columns are sparse (values at ~2 s epochs, NaN "
                     "between). marker column holds trigger labels at the "
                     "nearest sample. filt_<bank>_<CH> columns: one set per "
                     "filter bank."),
        }

        saved = save_recording(
            ts, data_save,
            filtered_banks=filtered_banks,
            sample_rate=self._reader.sample_rate,
            metric_history=metric_h,
            marker_events=(marker_events
                           if inc.get("markers", True) else None),
            channel_names=ch_names,
            meta=meta,
            ppg_timestamps=ppg_ts if ppg_ok else None,
            ppg_data=ppg_data      if ppg_ok else None,
            save_path=s.get("path"),
            formats=s.get("formats", {"csv", "edf"}),
            threshold_flag=inc.get("threshold_flag", True),
            thresholds=self._thr,
            record_meta=rec_meta,
            epoch_pad_s=float(self._settings.get("epoch_pad_s", 0.5)),
        )
        n_ppg = ppg_data.shape[0] if ppg_ok else 0
        extra = ""
        if "epochs" in saved:
            extra = f"\nEpochs: {Path(saved['epochs']).name}"
        prefix = "✓ Auto-saved" if autosaved else "✓ Saved"
        self._save_status.setText(
            f"{prefix} — EEG: {data_save.shape[0]}  PPG: {n_ppg}{extra}\n"
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
        self._session       = None
        self._recorder.start_recording()   # reset chunk lists; start unused
        self._recorder.stop_recording()
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

    def _on_edit_markers(self):
        from src.gui.marker_dialog import MarkerDialog
        dlg = MarkerDialog(self._marker_defs, self._marker_events, self)
        if dlg.exec():
            defs, events = dlg.result_data()
            self._marker_defs = defs
            self._marker_events = events
            self._rebuild_marker_buttons()
            self._settings["marker_defs"] = list(defs)
            save_settings(self._settings)
            self._marker_status.setText(
                f"{len(defs)} marker(s), {len(events)} event(s) recorded\n"
                "(also keys 1–9)")

    def _fire_marker(self, label: str):
        t = self._reader.now()
        self._marker_events.append((t, label))
        self._marker_status.setText(
            f"Last: {label} @ {time.strftime('%H:%M:%S')}"
            f"  (total {len(self._marker_events)})"
        )
        if self._lsl_pub is not None:
            self._lsl_pub.push_marker(label)

    def _refresh_bank_ui(self):
        """Sync the bank combo + summary label with the processor."""
        self._bank_combo.blockSignals(True)
        self._bank_combo.clear()
        self._bank_combo.addItems(self._processor.bank_names())
        self._bank_combo.setCurrentText(self._processor.active_bank)
        self._bank_combo.blockSignals(False)
        bank = self._processor._banks.get(self._processor.active_bank)
        if bank is not None and bank.stages:
            desc = " → ".join(s.describe() for s in bank.stages)
        else:
            desc = "no stages (passthrough)"
        self._bank_summary.setText(
            f"{len(self._processor.bank_names())} bank(s) — "
            f"{self._processor.active_bank}: {desc}")

    def _on_bank_changed(self, name: str):
        try:
            self._processor.set_active_bank(name)
        except KeyError:
            return
        self._filter_states = self._processor.make_filter_states(_N_CH)
        self._refresh_bank_ui()
        self._settings["active_bank"] = name
        save_settings(self._settings)

    def _on_manage_banks(self):
        from src.gui.filter_dialog import FilterBanksDialog
        from src.processing import FilterBank, FilterStage
        banks = [FilterBank(b.name, [FilterStage(**vars(s))
                                     for s in b.stages])
                 for b in self._processor.banks.values()]
        dlg = FilterBanksDialog(banks, self._processor.active_bank, self)
        if dlg.exec():
            new_banks, active = dlg.result_banks()
            try:
                self._processor.load_banks(
                    [b.to_dict() for b in new_banks], active)
            except Exception as exc:
                self._save_status.setText(f"⚠ Filter error: {exc}")
                return
            self._filter_states = self._processor.make_filter_states(_N_CH)
            self._settings["filter_banks"] = self._processor.bank_to_dicts()
            self._settings["active_bank"] = self._processor.active_bank
            save_settings(self._settings)
            self._refresh_bank_ui()

    def _on_app_settings(self):
        from src.gui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self._settings, self._processor, self)
        if dlg.exec():
            self._settings = dlg.settings()
            save_settings(self._settings)
            self._apply_settings()

    def _apply_settings(self):
        """Push persisted settings into live state (after Settings dialog)."""
        scale = float(self._settings.get("ui_scale", 1.0))
        if scale != self._ui_scale:
            self._apply_ui_scale(scale)
        self._thr = dict(self._settings.get("metric_thresholds", {}))
        self._mental_panel.set_thresholds(self._thr)
        self._marker_defs = list(self._settings.get(
            "marker_defs", self._marker_defs))
        self._rebuild_marker_buttons()

    def _apply_ui_scale(self, scale: float):
        self._ui_scale = scale
        app = QApplication.instance()
        if app is not None:
            app.setStyleSheet(build_stylesheet(scale))
        self._splitter.setSizes([int(230 * scale), int(1400 * scale)])
        self.resize(int(1440 * scale), int(860 * scale))

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
            new_ppg, pts = self._reader.get_new_ppg_samples(self._ppg_last_seen)
            self._ppg_last_seen += new_ppg.shape[0]
            if self._is_recording:
                pchunk, pcts = self._reader.get_new_ppg_samples(
                    self._rec_ppg_seen)
                self._rec_ppg_seen += pchunk.shape[0]
                self._recorder.append_ppg(pchunk, pcts)
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
        # Don't lose an in-progress recording on window close
        if self._is_recording:
            try:
                self._recorder.stop_recording()
                self._session = self._recorder.result()
                self._is_recording = False
                self._auto_save()
            except Exception:
                traceback.print_exc()
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
