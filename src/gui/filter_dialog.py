"""
filter_dialog.py
Filter-bank manager: named, ordered chains of filter stages
(bandpass / notch / highpass / lowpass) with presets and quick band add.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QHBoxLayout,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QSplitter, QVBoxLayout, QWidget, QCheckBox,
)

from src.processing import (
    BAND_PRESETS, FILTER_PRESETS, STAGE_KINDS, FilterBank, FilterStage,
)


class FilterBanksDialog(QDialog):
    """
    Edit the processor's filter banks.

    result() -> (banks: list[FilterBank], active_name: str)
    """

    def __init__(self, banks: list[FilterBank], active: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Filter Banks")
        self.resize(760, 480)
        self._banks = {b.name: FilterBank(b.name, [FilterStage(**vars(s))
                                                 for s in b.stages])
                       for b in banks}
        self._active = active if active in self._banks else next(iter(self._banks))
        self._build_ui()
        self._refresh_bank_list()

    # ── UI ────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        root = QHBoxLayout(self)

        # ── Left: bank list ────────────────────────────────────────────────
        left = QWidget()
        ll = QVBoxLayout(left)
        ll.setContentsMargins(0, 0, 0, 0)
        ll.addWidget(QLabel("Banks (active = display filter):"))
        self._bank_list = QListWidget()
        self._bank_list.setEditTriggers(
            QListWidget.EditTrigger.DoubleClicked |
            QListWidget.EditTrigger.EditKeyPressed)
        self._bank_list.currentRowChanged.connect(self._on_bank_selected)
        self._bank_list.itemChanged.connect(self._on_bank_renamed)
        ll.addWidget(self._bank_list, stretch=1)

        btn_row = QHBoxLayout()
        for text, fn in (("+ Add", self._add_bank),
                         ("⧉ Dup", self._dup_bank),
                         ("− Del", self._del_bank)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            btn_row.addWidget(b)
        ll.addLayout(btn_row)

        # ── Right: stages of selected bank ─────────────────────────────────
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.setContentsMargins(0, 0, 0, 0)
        self._bank_title = QLabel("")
        rl.addWidget(self._bank_title)

        self._stage_list = QListWidget()
        self._stage_list.currentRowChanged.connect(self._on_stage_selected)
        rl.addWidget(self._stage_list, stretch=1)

        srow = QHBoxLayout()
        for text, fn in (("▲ Up", lambda: self._move_stage(-1)),
                         ("▼ Down", lambda: self._move_stage(1)),
                         ("✕ Remove", self._remove_stage)):
            b = QPushButton(text)
            b.clicked.connect(fn)
            srow.addWidget(b)
        rl.addLayout(srow)

        # ── Stage editor ───────────────────────────────────────────────────
        ed = QHBoxLayout()
        self._ed_kind = QComboBox()
        self._ed_kind.addItems(STAGE_KINDS)
        self._ed_kind.currentIndexChanged.connect(self._sync_editor_fields)
        ed.addWidget(self._ed_kind)

        self._ed_low = QDoubleSpinBox();  self._ed_low.setRange(0.01, 120); self._ed_low.setDecimals(2); self._ed_low.setPrefix("lo ")
        self._ed_high = QDoubleSpinBox(); self._ed_high.setRange(0.01, 120); self._ed_high.setDecimals(2); self._ed_high.setPrefix("hi ")
        self._ed_freq = QDoubleSpinBox(); self._ed_freq.setRange(0.01, 120); self._ed_freq.setDecimals(2); self._ed_freq.setPrefix("f ")
        self._ed_enabled = QCheckBox("on")
        ed.addWidget(self._ed_low); ed.addWidget(self._ed_high); ed.addWidget(self._ed_freq)
        ed.addWidget(self._ed_enabled)
        rl.addLayout(ed)

        arow = QHBoxLayout()
        b_apply = QPushButton("Apply to selected")
        b_apply.clicked.connect(self._apply_stage)
        b_add = QPushButton("+ Add stage")
        b_add.clicked.connect(self._add_stage)
        arow.addWidget(b_apply); arow.addWidget(b_add)

        # Quick band add
        for band, (lo, hi) in BAND_PRESETS.items():
            b = QPushButton(band)
            b.setToolTip(f"Add bandpass {lo:g}–{hi:g} Hz stage")
            b.clicked.connect(lambda _c, l=lo, h=hi: self._quick_band(l, h))
            arow.addWidget(b)
        rl.addLayout(arow)

        # Presets
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Preset:"))
        self._preset_combo = QComboBox()
        self._preset_combo.addItems(["— load preset into selected bank —"]
                                    + list(FILTER_PRESETS.keys()))
        b_preset = QPushButton("Load")
        b_preset.clicked.connect(self._load_preset)
        prow.addWidget(self._preset_combo, stretch=1)
        prow.addWidget(b_preset)
        rl.addLayout(prow)

        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(left); split.addWidget(right)
        split.setSizes([220, 520])
        root.addWidget(split)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        rl.addWidget(btns)

    def _sync_editor_fields(self):
        kind = self._ed_kind.currentText()
        self._ed_low.setVisible(kind in ("bandpass", "highpass"))
        self._ed_high.setVisible(kind in ("bandpass", "lowpass"))
        self._ed_freq.setVisible(kind == "notch")

    # ── Bank list ─────────────────────────────────────────────────────────────

    def _refresh_bank_list(self):
        self._bank_list.blockSignals(True)
        self._bank_list.clear()
        for name in self._banks:
            item = QListWidgetItem(name)
            item.setData(Qt.ItemDataRole.UserRole, name)
            self._bank_list.addItem(item)
        for i in range(self._bank_list.count()):
            if self._bank_list.item(i).text() == self._active:
                self._bank_list.setCurrentRow(i)
                break
        self._bank_list.blockSignals(False)
        self._on_bank_selected(self._bank_list.currentRow())

    def _current_bank(self) -> FilterBank | None:
        item = self._bank_list.currentItem()
        if not item:
            return None
        name = item.data(Qt.ItemDataRole.UserRole) or item.text()
        return self._banks.get(name)

    def _on_bank_selected(self, _row: int):
        bank = self._current_bank()
        if bank is None:
            return
        self._bank_title.setText(
            f"<b>{bank.name}</b> — {len(bank.stages)} stage(s); "
            "order = application order")
        self._stage_list.clear()
        for s in bank.stages:
            QListWidgetItem(s.describe(), self._stage_list)
        if self._stage_list.count():
            self._stage_list.setCurrentRow(0)

    def _on_bank_renamed(self, item: QListWidgetItem):
        old = item.data(Qt.ItemDataRole.UserRole)
        new = item.text().strip()
        if not new or new == old or new in self._banks:
            self._refresh_bank_list()
            return
        bank = self._banks.pop(old)
        bank.name = new
        item.setData(Qt.ItemDataRole.UserRole, new)
        self._banks[new] = bank
        if self._active == old:
            self._active = new

    def _add_bank(self):
        name = f"Bank {len(self._banks) + 1}"
        while name in self._banks:
            name += "x"
        self._banks[name] = FilterBank(name, [])
        self._refresh_bank_list()
        for i in range(self._bank_list.count()):
            if self._bank_list.item(i).text() == name:
                self._bank_list.setCurrentRow(i)
                break

    def _dup_bank(self):
        bank = self._current_bank()
        if bank is None:
            return
        name = f"{bank.name} copy"
        while name in self._banks:
            name += "x"
        self._banks[name] = FilterBank(
            name, [FilterStage(**vars(s)) for s in bank.stages])
        self._refresh_bank_list()
        for i in range(self._bank_list.count()):
            if self._bank_list.item(i).text() == name:
                self._bank_list.setCurrentRow(i)
                break

    def _del_bank(self):
        bank = self._current_bank()
        if bank is None:
            return
        if len(self._banks) <= 1:
            QMessageBox.warning(self, "Filter Banks",
                                "At least one bank must remain.")
            return
        del self._banks[bank.name]
        if self._active == bank.name:
            self._active = next(iter(self._banks))
        self._refresh_bank_list()

    # ── Stage editing ─────────────────────────────────────────────────────────

    def _selected_stage(self):
        bank = self._current_bank()
        row = self._stage_list.currentRow()
        if bank is None or row < 0 or row >= len(bank.stages):
            return None, None
        return bank, bank.stages[row]

    def _on_stage_selected(self, _row: int):
        _bank, stage = self._selected_stage()
        if stage is None:
            return
        idx = STAGE_KINDS.index(stage.kind) if stage.kind in STAGE_KINDS else 0
        self._ed_kind.setCurrentIndex(idx)
        self._ed_low.setValue(stage.low)
        self._ed_high.setValue(stage.high)
        self._ed_freq.setValue(stage.freq)
        self._ed_enabled.setChecked(stage.enabled)
        self._sync_editor_fields()

    def _stage_from_editor(self) -> FilterStage:
        kind = self._ed_kind.currentText()
        return FilterStage(
            kind=kind,
            low=self._ed_low.value(),
            high=self._ed_high.value(),
            freq=self._ed_freq.value(),
            enabled=self._ed_enabled.isChecked(),
        )

    def _apply_stage(self):
        bank, stage = self._selected_stage()
        if bank is None or stage is None:
            return
        row = self._stage_list.currentRow()
        new = self._stage_from_editor()
        try:
            from src.processing import _stage_sos
            _stage_sos(new, 256.0)      # validate early
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid stage", str(exc))
            return
        bank.stages[row] = new
        self._stage_list.item(row).setText(new.describe())

    def _add_stage(self):
        bank = self._current_bank()
        if bank is None:
            return
        stage = self._stage_from_editor()
        bank.stages.append(stage)
        QListWidgetItem(stage.describe(), self._stage_list)
        self._stage_list.setCurrentRow(self._stage_list.count() - 1)

    def _quick_band(self, lo: float, hi: float):
        bank = self._current_bank()
        if bank is None:
            return
        stage = FilterStage("bandpass", low=lo, high=hi)
        bank.stages.append(stage)
        QListWidgetItem(stage.describe(), self._stage_list)
        self._stage_list.setCurrentRow(self._stage_list.count() - 1)

    def _move_stage(self, delta: int):
        bank = self._current_bank()
        row = self._stage_list.currentRow()
        if bank is None or row < 0:
            return
        new = row + delta
        if not (0 <= new < len(bank.stages)):
            return
        bank.stages[row], bank.stages[new] = bank.stages[new], bank.stages[row]
        texts = [s.describe() for s in bank.stages]
        self._stage_list.clear()
        for t in texts:
            QListWidgetItem(t, self._stage_list)
        self._stage_list.setCurrentRow(new)

    def _remove_stage(self):
        bank = self._current_bank()
        row = self._stage_list.currentRow()
        if bank is None or row < 0:
            return
        del bank.stages[row]
        self._stage_list.takeItem(row)
        if self._stage_list.count():
            self._stage_list.setCurrentRow(min(row, self._stage_list.count() - 1))

    def _load_preset(self):
        bank = self._current_bank()
        name = self._preset_combo.currentText()
        if bank is None or name not in FILTER_PRESETS:
            return
        bank.stages = [FilterStage.from_dict(d) for d in FILTER_PRESETS[name]]
        self._stage_list.clear()
        for s in bank.stages:
            QListWidgetItem(s.describe(), self._stage_list)
        if self._stage_list.count():
            self._stage_list.setCurrentRow(0)

    # ── Result ────────────────────────────────────────────────────────────────

    def _accept(self):
        # Validate every enabled stage compiles before accepting
        from src.processing import _stage_sos
        for bank in self._banks.values():
            for s in bank.enabled_stages():
                try:
                    _stage_sos(s, 256.0)
                except ValueError as exc:
                    QMessageBox.warning(
                        self, "Invalid filter",
                        f"Bank '{bank.name}': {exc}")
                    return
        self.accept()

    def result_banks(self) -> tuple[list[FilterBank], str]:
        return list(self._banks.values()), self._active
