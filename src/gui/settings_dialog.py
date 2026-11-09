"""
settings_dialog.py
Application settings: UI scale, autosave, epoch padding, per-metric
thresholds, and entry point to the filter-bank manager.
"""

from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QGroupBox, QHBoxLayout, QLabel, QMessageBox, QPushButton,
    QVBoxLayout, QWidget,
)

from src.settings import DEFAULT_SETTINGS, UI_SCALE_CHOICES
from src.gui.filter_dialog import FilterBanksDialog


class SettingsDialog(QDialog):
    """
    Modal app-settings dialog.

    settings() -> updated settings dict (caller persists + applies).
    `processor` is needed for the filter-bank manager.
    """

    def __init__(self, settings: dict, processor, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.resize(520, 560)
        self._s = {k: (dict(v) if isinstance(v, dict) else v)
                   for k, v in settings.items()}
        self._s["metric_thresholds"] = dict(
            settings.get("metric_thresholds", {}))
        self._processor = processor
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── General ────────────────────────────────────────────────────────
        gen = QGroupBox("General")
        gl = QVBoxLayout(gen)

        row = QHBoxLayout()
        row.addWidget(QLabel("UI scale:"))
        self._scale = QComboBox()
        for s in UI_SCALE_CHOICES:
            self._scale.addItem(f"{int(s * 100)}%", s)
        cur = self._s.get("ui_scale", 1.0)
        idx = self._scale.findData(cur)
        self._scale.setCurrentIndex(idx if idx >= 0 else 1)
        row.addWidget(self._scale)
        row.addStretch()
        gl.addLayout(row)

        self._autosave = QCheckBox(
            "Auto-save recording when Stop is pressed "
            "(Save ⚙ still available for re-export)")
        self._autosave.setChecked(self._s.get("autosave", True))
        gl.addWidget(self._autosave)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Epoch padding (s):"))
        self._pad = QDoubleSpinBox()
        self._pad.setRange(0.0, 5.0)
        self._pad.setSingleStep(0.25)
        self._pad.setValue(float(self._s.get("epoch_pad_s", 0.5)))
        row2.addWidget(self._pad)
        row2.addWidget(QLabel("  (padding around threshold-exceeding epochs)"))
        row2.addStretch()
        gl.addLayout(row2)
        root.addWidget(gen)

        # ── Metrics ────────────────────────────────────────────────────────
        met = QGroupBox("Mental-state thresholds (0–100)")
        ml = QVBoxLayout(met)
        ml.addWidget(QLabel(
            "Value at/above which a metric is flagged 'high' "
            "(gauge colour, above_threshold column, epoch extraction):"))
        th = self._s.get("metric_thresholds",
                         DEFAULT_SETTINGS["metric_thresholds"])
        self._thr_spins = {}
        for m in ("workload", "fatigue", "stress", "attention", "faa"):
            r = QHBoxLayout()
            r.addWidget(QLabel(m.capitalize() + ":"))
            sp = QDoubleSpinBox()
            sp.setRange(1.0, 100.0)
            sp.setDecimals(1)
            sp.setValue(float(th.get(m, 75.0)))
            self._thr_spins[m] = sp
            r.addWidget(sp)
            r.addStretch()
            ml.addLayout(r)
        root.addWidget(met)

        # ── Filters ────────────────────────────────────────────────────────
        flt = QGroupBox("Filters")
        fl = QVBoxLayout(flt)
        names = ", ".join(self._processor.bank_names()) or "—"
        fl.addWidget(QLabel(f"Banks: {names}   (active: "
                            f"{self._processor.active_bank})"))
        b_flt = QPushButton("Manage filter banks…")
        b_flt.clicked.connect(self._manage_filters)
        fl.addWidget(b_flt)
        root.addWidget(flt)

        # ── Danger zone ────────────────────────────────────────────────────
        dz = QHBoxLayout()
        b_reset = QPushButton("Restore all defaults")
        b_reset.clicked.connect(self._reset_defaults)
        dz.addWidget(b_reset)
        dz.addStretch()
        root.addLayout(dz)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── Slots ────────────────────────────────────────────────────────────────

    def _manage_filters(self):
        from src.processing import FilterBank, FilterStage
        banks = [FilterBank(b.name, [FilterStage(**vars(s))
                                     for s in b.stages])
                 for b in self._processor.banks.values()]
        dlg = FilterBanksDialog(banks, self._processor.active_bank, self)
        if dlg.exec():
            new_banks, active = dlg.result_banks()
            self._processor.load_banks(
                [b.to_dict() for b in new_banks], active)
            self._s["filter_banks"] = [b.to_dict() for b in new_banks]
            self._s["active_bank"] = active

    def _reset_defaults(self):
        if QMessageBox.question(
                self, "Restore defaults",
                "Reset UI scale, autosave, thresholds, filters and markers "
                "to factory defaults?") != QMessageBox.StandardButton.Yes:
            return
        d = DEFAULT_SETTINGS
        self._scale.setCurrentIndex(
            self._scale.findData(d["ui_scale"]))
        self._autosave.setChecked(d["autosave"])
        self._pad.setValue(d["epoch_pad_s"])
        for m, sp in self._thr_spins.items():
            sp.setValue(d["metric_thresholds"].get(m, 75.0))
        # filters → single default Cleaning bank
        from src.processing import FilterBank, FilterStage
        self._processor.load_banks(
            [{"name": "Cleaning",
              "stages": [
                  {"kind": "bandpass", "low": 0.5, "high": 45.0,
                   "freq": 50.0, "enabled": True},
                  {"kind": "notch", "low": 0.5, "high": 45.0,
                   "freq": 50.0, "enabled": True}]}],
            "Cleaning")
        self._s["filter_banks"] = self._processor.bank_to_dicts()
        self._s["active_bank"] = "Cleaning"
        self._s["marker_defs"] = list(d["marker_defs"])

    def _accept(self):
        self._s["ui_scale"] = float(self._scale.currentData())
        self._s["autosave"] = self._autosave.isChecked()
        self._s["epoch_pad_s"] = float(self._pad.value())
        self._s["metric_thresholds"] = {
            m: float(sp.value()) for m, sp in self._thr_spins.items()
        }
        self.accept()

    def settings(self) -> dict:
        return self._s
