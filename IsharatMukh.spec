# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for IsharatMukh — Muse S / LSL EEG dashboard (PyQt6).
# Build:  .venv/Scripts/python.exe -m PyInstaller --noconfirm IsharatMukh.spec

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas, binaries, hiddenimports = [], [], []

for pkg in ("pylsl", "muselsl", "bleak", "mne", "pyqtgraph", "pyedflib",
            "pygatt", "bitstring", "serial"):
    d, b, h = collect_all(pkg)
    datas += d; binaries += b; hiddenimports += h

hiddenimports += collect_submodules("scipy.signal")
hiddenimports += ["PyQt6.QtNetwork", "winrt", "asyncio"]

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["PyQt5", "PySide2", "PySide6", "tkinter", "matplotlib.tests"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="IsharatMukh",
    console=False,          # GUI-only; flip to True for a debug build
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="IsharatMukh",
)
