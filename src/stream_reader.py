"""
stream_reader.py
Background thread that reads the Muse S LSL stream and fills a thread-safe
circular buffer. Provides a snapshot of the latest N seconds on demand.
"""

import threading
import time
from collections import deque

import numpy as np

try:
    from pylsl import StreamInlet, resolve_byprop, StreamInfo
    LSL_AVAILABLE = True
except ImportError:
    LSL_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
MUSE_SAMPLE_RATE = 256          # Hz — Muse S native sample rate
CHANNEL_NAMES    = ["TP9", "AF7", "AF8", "TP10", "AUX"]
CHANNEL_COUNT    = 5
BUFFER_SECONDS   = 60           # keep last 60 s in memory


class StreamReader:
    """
    Connects to a running LSL EEG stream (started via muselsl or BlueMuse)
    and continuously fills a circular deque buffer in a daemon thread.

    Usage
    -----
    reader = StreamReader()
    reader.start()
    ...
    raw, ts = reader.get_window(seconds=5)   # (n_samples, 5) + timestamps
    reader.stop()
    """

    def __init__(self, buffer_seconds: int = BUFFER_SECONDS):
        self._max_samples = buffer_seconds * MUSE_SAMPLE_RATE
        # Each element: (timestamp, [ch0, ch1, ch2, ch3, ch4])
        self._buffer: deque = deque(maxlen=self._max_samples)
        self._lock   = threading.Lock()
        self._thread: threading.Thread | None = None
        self._running = False
        self._connected = False
        self._inlet: "StreamInlet | None" = None
        self.sample_rate: float = float(MUSE_SAMPLE_RATE)
        self.dropped_samples: int = 0
        self._demo_mode = False          # activated when no real stream found

    # ── Public API ──────────────────────────────────────────────────────────────

    def start(self, timeout: float = 10.0) -> bool:
        """Resolve the LSL stream and start the background thread."""
        if self._running:
            return self._connected

        if not LSL_AVAILABLE:
            print("[StreamReader] pylsl not installed — falling back to demo mode.")
            self._start_demo()
            return False

        print("[StreamReader] Searching for EEG LSL stream …")
        streams = resolve_byprop("type", "EEG", timeout=timeout)

        if not streams:
            print("[StreamReader] No LSL stream found — starting demo mode.")
            self._start_demo()
            return False

        self._inlet = StreamInlet(streams[0], max_buflen=60)
        info = self._inlet.info()
        self.sample_rate = info.nominal_srate() or MUSE_SAMPLE_RATE
        print(f"[StreamReader] Connected — {info.name()} @ {self.sample_rate:.0f} Hz")
        self._connected = True
        self._running   = True
        self._thread    = threading.Thread(target=self._read_loop, daemon=True, name="lsl-reader")
        self._thread.start()
        return True

    def stop(self):
        """Signal the background thread to stop."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self._inlet:
            self._inlet.close_stream()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_demo(self) -> bool:
        return self._demo_mode

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_window(self, seconds: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        """
        Return the latest `seconds` of data.

        Returns
        -------
        data : ndarray, shape (n_samples, 5)  — one column per channel
        timestamps : ndarray, shape (n_samples,)
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
        """Return the entire buffer contents."""
        with self._lock:
            items = list(self._buffer)
        if not items:
            return np.empty((0, CHANNEL_COUNT)), np.empty(0)
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    # ── Internal ────────────────────────────────────────────────────────────────

    def _read_loop(self):
        """Continuously pull samples from the LSL inlet."""
        while self._running:
            try:
                sample, timestamp = self._inlet.pull_sample(timeout=1.0)
                if sample is not None:
                    # Pad to 5 channels if AUX is absent
                    if len(sample) < CHANNEL_COUNT:
                        sample = list(sample) + [0.0] * (CHANNEL_COUNT - len(sample))
                    with self._lock:
                        self._buffer.append((timestamp, sample[:CHANNEL_COUNT]))
            except Exception as exc:
                if self._running:
                    print(f"[StreamReader] Read error: {exc}")
                    time.sleep(0.1)

    # ── Demo mode ───────────────────────────────────────────────────────────────

    def _start_demo(self):
        """Simulate a Muse stream with synthetic EEG for testing the UI."""
        self._demo_mode = True
        self._running   = True
        self._thread    = threading.Thread(target=self._demo_loop, daemon=True, name="demo-reader")
        self._thread.start()

    def _demo_loop(self):
        """Generate synthetic EEG data at 256 Hz."""
        t = 0.0
        dt = 1.0 / MUSE_SAMPLE_RATE
        rng = np.random.default_rng(seed=42)
        while self._running:
            sample = self._make_synthetic_sample(t, rng)
            with self._lock:
                self._buffer.append((time.time(), sample))
            t += dt
            time.sleep(dt)

    @staticmethod
    def _make_synthetic_sample(t: float, rng) -> list[float]:
        """Produce one plausible fake EEG sample (µV)."""
        alpha = 20.0 * np.sin(2 * np.pi * 10 * t)          # ~10 Hz alpha
        theta = 10.0 * np.sin(2 * np.pi * 6  * t)           # ~6 Hz theta
        noise = rng.normal(0, 5, CHANNEL_COUNT)
        base  = alpha + theta
        return [base + noise[i] for i in range(CHANNEL_COUNT)]
