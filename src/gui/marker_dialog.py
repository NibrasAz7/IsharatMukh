"""
marker_dialog.py
Marker designer: edit trigger definitions (rename, drag-reorder, add/remove)
and edit already-fired marker events before saving.
"""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel,
    QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)


class MarkerDialog(QDialog):
    """
    Edit marker definitions + fired events.

    result() -> (defs: list[str], events: list[(t, label)])
    """

    def __init__(self, defs: list[str],
                 events: list[tuple[float, str]], parent=None):
        super().__init__(parent)
        self.setWindowTitle("Markers / Triggers")
        self.resize(560, 520)
        self._defs = list(defs)
        self._events = list(events)     # [(unix_ts, label)]
        self._build_ui()

    def _build_ui(self):
        root = QVBoxLayout(self)

        # ── Definitions ─────────────────────────────────────────────────────
        root.addWidget(QLabel(
            "<b>Marker buttons</b> — double-click to rename, drag to reorder "
            "(order = shortcut keys 1–9):"))
        self._def_list = QListWidget()
        self._def_list.setDragDropMode(QListWidget.DragDropMode.InternalMove)
        self._def_list.setDefaultDropAction(Qt.DropAction.MoveAction)
        self._def_list.setEditTriggers(
            QListWidget.EditTrigger.DoubleClicked |
            QListWidget.EditTrigger.EditKeyPressed)
        for d in self._defs:
            QListWidgetItem(d, self._def_list)
        root.addWidget(self._def_list, stretch=1)

        drow = QHBoxLayout()
        b_add = QPushButton("+ Add")
        b_del = QPushButton("− Remove")
        b_add.clicked.connect(self._add_def)
        b_del.clicked.connect(self._del_def)
        drow.addWidget(b_add); drow.addWidget(b_del); drow.addStretch()
        root.addLayout(drow)

        # ── Fired events ───────────────────────────────────────────────────
        root.addWidget(QLabel(
            "<b>Fired events</b> — relabel or delete before saving "
            "(timestamps are read-only):"))
        self._ev_table = QTableWidget(0, 3)
        self._ev_table.setHorizontalHeaderLabels(["Time", "Label", ""])
        self._ev_table.horizontalHeader().setStretchLastSection(False)
        self._ev_table.setColumnWidth(0, 110)
        self._ev_table.setColumnWidth(1, 160)
        self._ev_table.setColumnWidth(2, 40)
        self._ev_table.verticalHeader().setVisible(False)
        for t, label in self._events:
            self._add_event_row(t, label)
        root.addWidget(self._ev_table, stretch=1)

        erow = QHBoxLayout()
        b_edel = QPushButton("Delete selected event")
        b_edel.clicked.connect(self._del_event)
        erow.addWidget(b_edel); erow.addStretch()
        root.addLayout(erow)

        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok |
                                QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        root.addWidget(btns)

    # ── Definitions ───────────────────────────────────────────────────────────

    def _add_def(self):
        item = QListWidgetItem("New marker", self._def_list)
        item.setFlags(item.flags() | Qt.ItemFlag.ItemIsEditable)
        self._def_list.editItem(item)

    def _del_def(self):
        row = self._def_list.currentRow()
        if row >= 0:
            self._def_list.takeItem(row)

    # ── Events ────────────────────────────────────────────────────────────────

    def _add_event_row(self, t: float, label: str):
        row = self._ev_table.rowCount()
        self._ev_table.insertRow(row)
        import time as _t
        ts_item = QTableWidgetItem(_t.strftime("%H:%M:%S", _t.localtime(t)))
        ts_item.setFlags(ts_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        ts_item.setData(Qt.ItemDataRole.UserRole, t)
        self._ev_table.setItem(row, 0, ts_item)

        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self._defs)
        combo.setCurrentText(label)
        self._ev_table.setCellWidget(row, 1, combo)

        b = QPushButton("✕")
        b.setFixedWidth(30)
        b.clicked.connect(lambda _c, r=row: self._ev_table.removeRow(r))
        self._ev_table.setCellWidget(row, 2, b)

    def _del_event(self):
        row = self._ev_table.currentRow()
        if row >= 0:
            self._ev_table.removeRow(row)

    # ── Result ────────────────────────────────────────────────────────────────

    def _accept(self):
        defs = []
        for i in range(self._def_list.count()):
            name = self._def_list.item(i).text().strip()
            if name and name not in defs:
                defs.append(name)
        if not defs:
            QMessageBox.warning(self, "Markers", "At least one marker is required.")
            return
        self._defs = defs
        self.accept()

    def result_data(self) -> tuple[list[str], list[tuple[float, str]]]:
        events = []
        for r in range(self._ev_table.rowCount()):
            t_item = self._ev_table.item(r, 0)
            combo = self._ev_table.cellWidget(r, 1)
            if t_item is None or combo is None:
                continue
            t = t_item.data(Qt.ItemDataRole.UserRole)
            label = combo.currentText().strip()
            if label:
                events.append((float(t), label))
        return self._defs, events
