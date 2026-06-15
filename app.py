"""
app.py
Entry point for the Muse S EEG Desktop Dashboard (PyQt6).

Usage
-----
    python app.py
"""

import sys

from PyQt6.QtWidgets import QApplication

from src.gui.main_window import MainWindow
from src.gui.theme import DARK_QSS


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(DARK_QSS)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
