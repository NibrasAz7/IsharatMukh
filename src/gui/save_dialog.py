"""
save_dialog.py
Dialog for configuring recording save options (path, format, channel selection).
"""

from pathlib import Path

import pyqtgraph as pg
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QFileDialog,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit,
    QPushButton, QVBoxLayout,
)

EEG_CHANNELS   = ["TP9", "AF7", "AF8", "TP10"]
HIGH_THRESHOLD = 75   # must match mental_panel.HIGH_THRESHOLD


class SaveSettingsDialog(QDialog):
    """
    Modal dialog for recording save settings.

    Usage
    -----
    dlg = SaveSettingsDialog(current_settings, parent=window)
    if dlg.exec():
        settings = dlg.settings
    """

    def __init__(self, settings: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Save Settings")
        self.setMinimumWidth(430)
        self._settings = dict(settings)   # working copy
        self._build_ui()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)
        layout.setContentsMargins(14, 14, 14, 14)

        # ── Save location ─────────────────────────────────────────────────────
        path_group = QGroupBox("Save Location")
        path_layout = QHBoxLayout(path_group)
        self._path_edit = QLineEdit(self._settings.get("path", ""))
        self._path_edit.setPlaceholderText("Directory where files will be written…")
        btn_browse = QPushButton("Browse…")
        btn_browse.setFixedWidth(75)
        btn_browse.clicked.connect(self._browse)
        path_layout.addWidget(self._path_edit)
        path_layout.addWidget(btn_browse)
        layout.addWidget(path_group)

        # ── File format ───────────────────────────────────────────────────────
        fmt_group  = QGroupBox("File Format")
        fmt_layout = QVBoxLayout(fmt_group)
        fmts = self._settings.get("formats", {"csv", "edf"})
        self._cb_csv = QCheckBox("CSV  (raw + filtered + metrics + markers)")
        self._cb_edf = QCheckBox("EDF+  (EEG + marker annotations — EEGLAB/MNE)")
        self._cb_fif = QCheckBox("FIF  (MNE-native, markers as annotations)")
        self._cb_csv.setChecked("csv" in fmts)
        self._cb_edf.setChecked("edf" in fmts)
        self._cb_fif.setChecked("fif" in fmts)
        fmt_layout.addWidget(self._cb_csv)
        fmt_layout.addWidget(self._cb_edf)
        fmt_layout.addWidget(self._cb_fif)
        layout.addWidget(fmt_group)

        # ── What to include ───────────────────────────────────────────────────
        inc_group  = QGroupBox("Include in Recording")
        inc_layout = QVBoxLayout(inc_group)
        inc = self._settings.get("include", {})

        # EEG channel row
        ch_row = QHBoxLayout()
        ch_row.addWidget(QLabel("EEG channels:"))
        ch_include = inc.get("eeg_ch", [True]*4)
        self._ch_boxes: list[QCheckBox] = []
        for i, name in enumerate(EEG_CHANNELS):
            cb = QCheckBox(name)
            cb.setChecked(ch_include[i] if i < len(ch_include) else True)
            self._ch_boxes.append(cb)
            ch_row.addWidget(cb)
        ch_row.addStretch()
        inc_layout.addLayout(ch_row)

        # Other include options
        self._cb_filtered = QCheckBox(
            "Filtered EEG columns  (filt_<CH> alongside raw_<CH>)"
        )
        self._cb_mental = QCheckBox(
            "Mental state metrics  (workload, fatigue, stress, attention, FAA)"
        )
        self._cb_markers = QCheckBox("Marker / trigger column + events")
        self._cb_ppg    = QCheckBox("PPG  (Ambient, IR, Red — separate CSV)")
        self._cb_thresh = QCheckBox(
            f"Threshold-exceeded flag  (1 when any metric ≥ {HIGH_THRESHOLD})"
        )
        self._cb_filtered.setChecked(inc.get("filtered", True))
        self._cb_mental.setChecked(inc.get("mental", True))
        self._cb_markers.setChecked(inc.get("markers", True))
        self._cb_ppg.setChecked(inc.get("ppg", True))
        self._cb_thresh.setChecked(inc.get("threshold_flag", True))
        self._cb_thresh.setEnabled(self._cb_mental.isChecked())
        self._cb_mental.toggled.connect(self._cb_thresh.setEnabled)

        inc_layout.addWidget(self._cb_filtered)
        inc_layout.addWidget(self._cb_mental)
        inc_layout.addWidget(self._cb_markers)
        inc_layout.addWidget(self._cb_ppg)
        inc_layout.addWidget(self._cb_thresh)
        layout.addWidget(inc_group)

        # ── Buttons ───────────────────────────────────────────────────────────
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    # ── Private ───────────────────────────────────────────────────────────────

    def _browse(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Save Directory", self._path_edit.text()
        )
        if directory:
            self._path_edit.setText(directory)

    def _accept(self):
        formats: set[str] = set()
        if self._cb_csv.isChecked():
            formats.add("csv")
        if self._cb_edf.isChecked():
            formats.add("edf")
        if self._cb_fif.isChecked():
            formats.add("fif")

        self._settings["path"]    = self._path_edit.text().strip()
        self._settings["formats"] = formats
        self._settings["include"] = {
            "eeg":            True,
            "eeg_ch":         [cb.isChecked() for cb in self._ch_boxes],
            "filtered":       self._cb_filtered.isChecked(),
            "mental":         self._cb_mental.isChecked(),
            "markers":        self._cb_markers.isChecked(),
            "ppg":            self._cb_ppg.isChecked(),
            "threshold_flag": self._cb_thresh.isChecked(),
        }
        self.accept()

    # ── Public ────────────────────────────────────────────────────────────────

    @property
    def settings(self) -> dict:
        return self._settings
