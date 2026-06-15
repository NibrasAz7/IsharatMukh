"""
stream_reader.py
Standalone BLE stream reader for Muse S.

Connects directly to the Muse S headset using muselsl + bleak (pure Python BLE).
No external tools (BlueMuse, muselsl CLI) required.

Architecture
------------
scan_devices()       →  finds nearby Muse headsets via BLE scan (background thread)
start(address)       →  connects to Muse via BLE, registers EEG callback
_on_eeg_data()       →  muselsl callback, writes samples into circular deque
get_window(seconds)  →  returns latest N seconds as numpy array
stop()               →  sends stop command, disconnects BLE
"""

import threading
import time
from collections import deque
from typing import Callable

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────────────
MUSE_SAMPLE_RATE = 256          # Hz
CHANNEL_NAMES    = ["TP9", "AF7", "AF8", "TP10", "AUX"]
CHANNEL_COUNT    = 5
BUFFER_SECONDS   = 60           # keep last 60 s in memory
SCAN_TIMEOUT     = 10.5         # seconds BLE scan runs


class StreamReader:
    """
    Standalone Muse S EEG reader using muselsl + bleak (pure Python BLE).

    Usage
    -----
    reader = StreamReader()

    # Option A: scan first
    reader.scan_devices(on_found=lambda devices: ...)
    reader.start(address="AA:BB:CC:DD:EE:FF")

    # Option B: connect to first available Muse automatically
    reader.start()

    raw, ts = reader.get_window(seconds=5)   # (n_samples, 5), timestamps
    reader.stop()
    """

    def __init__(self, buffer_seconds: int = BUFFER_SECONDS):
        self._max_samples = buffer_seconds * MUSE_SAMPLE_RATE
        self._buffer: deque = deque(maxlen=self._max_samples)
        self._lock    = threading.Lock()

        self._muse            = None        # muselsl.Muse instance
        self._stream_thread: threading.Thread | None = None
        self._scan_thread:   threading.Thread | None = None
        self._stop_event      = threading.Event()

        self._running   = False
        self._connected = False
        self._demo_mode = False
        self._scanning  = False

        self.sample_rate: float = float(MUSE_SAMPLE_RATE)
        self.device_name: str   = ""
        self.device_address: str = ""

        # List of discovered devices: [{"name": ..., "address": ...}, ...]
        self.discovered_devices: list[dict] = []

    # ── Scanning ─────────────────────────────────────────────────────────────────

    def scan_devices(self, on_found: Callable[[list], None] | None = None,
                     timeout: float = SCAN_TIMEOUT) -> None:
        """
        Scan for nearby Muse headsets in a background thread.
        Calls `on_found(devices)` when done, where devices is a list of
        {"name": str, "address": str} dicts.
        """
        if self._scanning:
            return
        self._scanning = True
        self.discovered_devices = []

        def _scan():
            try:
                from muselsl.stream import list_muses
                print(f"[StreamReader] Scanning for Muse devices ({timeout:.0f}s)…")
                devices = list_muses(backend="bleak") or []
                self.discovered_devices = devices
                print(f"[StreamReader] Found {len(devices)} device(s): "
                      f"{[d['name'] for d in devices]}")
            except Exception as exc:
                print(f"[StreamReader] Scan error: {exc}")
                self.discovered_devices = []
            finally:
                self._scanning = False
                if on_found:
                    on_found(self.discovered_devices)

        self._scan_thread = threading.Thread(target=_scan, daemon=True, name="ble-scan")
        self._scan_thread.start()

    # ── Connect & stream ─────────────────────────────────────────────────────────

    def start(self, address: str | None = None, name: str | None = None) -> bool:
        """
        Connect to a Muse S and start streaming EEG into the buffer.

        Parameters
        ----------
        address : BLE MAC address (e.g. "AA:BB:CC:DD:EE:FF").
                  If None, auto-scans and picks the first Muse found.
        name    : Device name filter (optional, used only when address is None).

        Returns True if BLE connection started (async), False on failure.
        Falls back to demo mode if muselsl is unavailable.
        """
        if self._running:
            return self._connected

        try:
            from muselsl.muse import Muse
            from muselsl.stream import find_muse
        except ImportError:
            print("[StreamReader] muselsl not installed — falling back to demo mode.")
            self._start_demo()
            return False

        self._stop_event.clear()

        def _connect_and_stream():
            nonlocal address, name
            try:
                # Auto-discover if no address given
                if not address:
                    print("[StreamReader] No address given — scanning for Muse…")
                    found = find_muse(name=name, backend="bleak")
                    if not found:
                        print("[StreamReader] No Muse found — starting demo mode.")
                        self._start_demo()
                        return
                    address = found["address"]
                    name    = found["name"]

                print(f"[StreamReader] Connecting to {name or 'Muse'} ({address})…")

                self._muse = Muse(
                    address=address,
                    name=name,
                    callback_eeg=self._on_eeg_data,
                    backend="bleak",
                )

                ok = self._muse.connect(retries=2)
                if not ok:
                    print("[StreamReader] BLE connect failed — falling back to demo mode.")
                    self._start_demo()
                    return

                self._muse.start()
                self._connected      = True
                self._running        = True
                self.device_address  = address
                self.device_name     = name or "Muse"
                print(f"[StreamReader] Streaming from {self.device_name}.")

                # Keep the BLE event loop alive until stop() is called
                from muselsl import backends
                while not self._stop_event.is_set():
                    backends.sleep(0.5)

                # Graceful disconnect
                try:
                    self._muse.stop()
                    self._muse.disconnect()
                except Exception:
                    pass

            except Exception as exc:
                print(f"[StreamReader] Connection error: {exc}")
                if not self._running:
                    self._start_demo()
            finally:
                self._connected = False
                self._running   = False

        self._stream_thread = threading.Thread(
            target=_connect_and_stream, daemon=True, name="muse-ble"
        )
        self._stream_thread.start()

        # Wait up to 15 s for connect to succeed before returning
        deadline = time.time() + 15.0
        while time.time() < deadline:
            if self._connected or self._demo_mode:
                break
            time.sleep(0.2)

        return self._connected

    def stop(self):
        """Disconnect from the Muse S and stop the background thread."""
        self._stop_event.set()
        self._running   = False
        self._connected = False
        if self._stream_thread:
            self._stream_thread.join(timeout=5.0)
        self._muse = None

    # ── Public data API ───────────────────────────────────────────────────────────

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    @property
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_window(self, seconds: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the latest `seconds` of buffered EEG.

        Returns
        -------
        data       : ndarray (n_samples, 5)  — µV per channel
        timestamps : ndarray (n_samples,)    — Unix timestamps
        """
        n = int(seconds * self.sample_rate)
        with self._lock:
            items = list(self._buffer)[-n:]
        if not items:
            return np.empty((0, CHANNEL_COUNT)), np.empty(0)
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the entire buffer (up to BUFFER_SECONDS)."""
        with self._lock:
            items = list(self._buffer)
        if not items:
            return np.empty((0, CHANNEL_COUNT)), np.empty(0)
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    # ── muselsl EEG callback ──────────────────────────────────────────────────────

    def _on_eeg_data(self, data: np.ndarray, timestamps: np.ndarray) -> None:
        """
        Called by muselsl on every BLE EEG notification.

        Parameters
        ----------
        data       : (n_channels, n_samples) — typically (5, 12)  µV
        timestamps : (n_samples,)            — Unix timestamps
        """
        # Transpose to (n_samples, n_channels) and pad/trim to CHANNEL_COUNT
        samples = data.T                         # (12, 5)
        n_ch = samples.shape[1]
        if n_ch < CHANNEL_COUNT:
            pad = np.zeros((samples.shape[0], CHANNEL_COUNT - n_ch))
            samples = np.hstack([samples, pad])
        elif n_ch > CHANNEL_COUNT:
            samples = samples[:, :CHANNEL_COUNT]

        now = time.time()
        with self._lock:
            for i, ts in enumerate(timestamps):
                self._buffer.append((ts if ts > 0 else now, samples[i].tolist()))

    # ── Demo mode ─────────────────────────────────────────────────────────────────

    def _start_demo(self):
        """Generate synthetic EEG when no real Muse is available."""
        self._demo_mode = True
        self._running   = True
        t = threading.Thread(target=self._demo_loop, daemon=True, name="demo-eeg")
        t.start()

    def _demo_loop(self):
        t = 0.0
        dt = 1.0 / MUSE_SAMPLE_RATE
        rng = np.random.default_rng(seed=42)
        while self._running and not self._stop_event.is_set():
            alpha = 20.0 * np.sin(2 * np.pi * 10 * t)
            theta = 10.0 * np.sin(2 * np.pi * 6  * t)
            noise = rng.normal(0, 5, CHANNEL_COUNT)
            sample = [alpha + theta + noise[i] for i in range(CHANNEL_COUNT)]
            with self._lock:
                self._buffer.append((time.time(), sample))
            t += dt
            time.sleep(dt)
        self._running   = False
        self._demo_mode = False
