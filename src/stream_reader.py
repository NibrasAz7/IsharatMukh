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

        self._running      = False
        self._connected    = False
        self._scanning     = False
        self._sample_count = 0    # monotonically increasing total samples written

        self.sample_rate: float   = float(MUSE_SAMPLE_RATE)
        self.device_name: str     = ""
        self.device_address: str  = ""
        self.last_error: str      = ""

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

        Returns True on successful BLE connection, False on failure.
        Never falls back to demo — buffer stays empty if no real device.
        """
        if self._running:
            return self._connected

        try:
            from muselsl.muse import Muse
            from muselsl.stream import find_muse
        except ImportError:
            print("[StreamReader] muselsl not installed.")
            self.last_error = "muselsl not installed"
            return False

        self._stop_event.clear()
        self.last_error = ""

        def _connect_and_stream():
            nonlocal address, name
            try:
                if not address:
                    print("[StreamReader] No address given — scanning for Muse…")
                    found = find_muse(name=name, backend="bleak")
                    if not found:
                        print("[StreamReader] No Muse found.")
                        self.last_error = "No Muse device found during scan"
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
                    print("[StreamReader] BLE connect failed.")
                    self.last_error = "BLE connection failed"
                    return

                self._muse.start()
                self._connected      = True
                self._running        = True
                self.device_address  = address
                self.device_name     = name or "Muse"
                print(f"[StreamReader] Streaming from {self.device_name}.")

                from muselsl import backends
                while not self._stop_event.is_set():
                    backends.sleep(0.5)

                try:
                    self._muse.stop()
                    self._muse.disconnect()
                except Exception:
                    pass

            except Exception as exc:
                print(f"[StreamReader] Connection error: {exc}")
                self.last_error = str(exc)
            finally:
                self._connected = False
                self._running   = False

        self._stream_thread = threading.Thread(
            target=_connect_and_stream, daemon=True, name="muse-ble"
        )
        self._stream_thread.start()

        # Wait up to 20 s for the BLE connection to complete
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if self._connected or self.last_error:
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
    def is_scanning(self) -> bool:
        return self._scanning

    @property
    def sample_count(self) -> int:
        """Total samples ever written to the buffer (monotonically increasing)."""
        return self._sample_count

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_new_samples(self, from_count: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Return only samples written after `from_count`.

        Returns
        -------
        data       : (n_new, 5)
        timestamps : (n_new,)
        """
        with self._lock:
            n_new = min(self._sample_count - from_count, len(self._buffer))
            if n_new <= 0:
                return np.empty((0, CHANNEL_COUNT)), np.empty(0)
            items = list(self._buffer)[-n_new:]
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

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
                self._sample_count += 1

