"""
lsl_io.py
Lab Streaming Layer I/O for the Muse S dashboard.

LSLReader  — duck-types StreamReader: connect to an existing LSL stream
             (e.g. BlueMuse, muselsl stream, OpenBCI GUI) as the EEG source.
LSLOutlet  — re-broadcast the app's EEG + trigger markers as LSL streams
             so LabRecorder or other tools can subscribe.
"""

import threading
from collections import deque
from typing import Callable

import numpy as np

CHANNEL_COUNT  = 5
BUFFER_SECONDS = 60


class LSLReader:
    """
    Read an existing LSL stream into a circular buffer.

    Mirrors the StreamReader API used by the GUI:
        resolve()                        → list available streams
        start(stream_info) / start_async → open inlet + pull thread
        get_new_samples / get_window / get_all
        stop() / clear()
        sample_count, is_connected, sample_rate, device_name, device_address

    Sample timestamps are LSL local_clock() values (not Unix time) — use
    now() when logging markers so they align with the stream's timebase.
    """

    def __init__(self, buffer_seconds: int = BUFFER_SECONDS):
        self._buf_seconds = buffer_seconds
        self._buffer: deque = deque(maxlen=buffer_seconds * 512)
        self._lock = threading.Lock()
        self._inlet = None
        self._pull_thread: threading.Thread | None = None
        self._stop = threading.Event()

        self._connected   = False
        self._sample_count = 0
        self._n_ch_stream = 0

        self.sample_rate     = 256.0
        self.ppg_sample_rate = 0.0
        self.device_name     = ""
        self.device_address  = ""
        self.last_error      = ""

    # ── Discovery ─────────────────────────────────────────────────────────────

    @staticmethod
    def resolve(timeout: float = 3.0) -> list[dict]:
        """
        List available LSL streams as device-list dicts.
        Each entry carries the StreamInfo under '_stream_info' for connect.
        """
        from pylsl import resolve_streams
        out = []
        for s in resolve_streams(wait_time=timeout):
            out.append({
                "name":    f"{s.name()} [{s.type() or '?'}]",
                "address": s.source_id() or s.name(),
                "rssi":    None,
                "n_ch":    s.channel_count(),
                "fs":      s.nominal_srate(),
                "_stream_info": s,
            })
        return out

    # ── Connect / stream ──────────────────────────────────────────────────────

    def start(self, stream_info=None) -> bool:
        """Open an LSL inlet and start the background pull thread."""
        if self._connected:
            return True
        self.last_error = ""
        try:
            from pylsl import StreamInlet, resolve_streams

            if stream_info is None:
                streams = resolve_streams(wait_time=3.0)
                eeg = [s for s in streams if (s.type() or "").upper()
                       in ("EEG", "EXG", "MUSE")] or streams
                if not eeg:
                    self.last_error = "No LSL stream found"
                    return False
                stream_info = eeg[0]

            self._inlet = StreamInlet(stream_info, max_buflen=360)
            self.sample_rate  = float(stream_info.nominal_srate()) or 256.0
            self._n_ch_stream = stream_info.channel_count()
            self.device_name    = stream_info.name()
            self.device_address = stream_info.source_id() or stream_info.name()
            self._buffer = deque(
                maxlen=int(self._buf_seconds * self.sample_rate)
            )

            self._stop.clear()
            self._pull_thread = threading.Thread(
                target=self._pull_loop, daemon=True, name="lsl-pull"
            )
            self._pull_thread.start()
            self._connected = True
            print(f"[LSLReader] Reading '{self.device_name}' "
                  f"({self._n_ch_stream} ch @ {self.sample_rate:g} Hz)")
            return True
        except Exception as exc:
            self.last_error = str(exc)
            print(f"[LSLReader] connect error: {exc}")
            return False

    def start_async(self, stream_info=None,
                    on_result: Callable[[bool, str], None] | None = None):
        def _run():
            ok = self.start(stream_info)
            if on_result:
                on_result(ok, self.last_error)
        threading.Thread(target=_run, daemon=True, name="lsl-async").start()

    def _pull_loop(self):
        while not self._stop.is_set():
            try:
                chunk, ts = self._inlet.pull_chunk(timeout=0.25)
            except Exception as exc:
                print(f"[LSLReader] pull error: {exc}")
                break
            if not ts:
                continue
            with self._lock:
                for i, t in enumerate(ts):
                    row = [float(v) for v in chunk[i][:CHANNEL_COUNT]]
                    if len(row) < CHANNEL_COUNT:
                        row += [0.0] * (CHANNEL_COUNT - len(row))
                    self._buffer.append((t, row))
                    self._sample_count += 1
        self._connected = False

    def stop(self):
        self._stop.set()
        self._connected = False
        if self._pull_thread:
            self._pull_thread.join(timeout=3.0)
        if self._inlet is not None:
            try:
                self._inlet.close_stream()
            except Exception:
                pass
        self._inlet = None

    def clear(self):
        with self._lock:
            self._buffer.clear()
            self._sample_count = 0

    # ── Public data API (mirrors StreamReader) ────────────────────────────────

    @staticmethod
    def now() -> float:
        """Current time on the LSL clock — use for markers on this source."""
        try:
            from pylsl import local_clock
            return local_clock()
        except Exception:
            import time
            return time.time()

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def sample_count(self) -> int:
        return self._sample_count

    @property
    def ppg_count(self) -> int:
        return 0

    @property
    def buffer_length(self) -> int:
        with self._lock:
            return len(self._buffer)

    def get_new_samples(self, from_count: int) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            n_new = min(self._sample_count - from_count, len(self._buffer))
            if n_new <= 0:
                return np.empty((0, CHANNEL_COUNT)), np.empty(0)
            items = list(self._buffer)[-n_new:]
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    def get_window(self, seconds: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
        n = int(seconds * self.sample_rate)
        with self._lock:
            items = list(self._buffer)[-n:]
        if not items:
            return np.empty((0, CHANNEL_COUNT)), np.empty(0)
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    def get_all(self) -> tuple[np.ndarray, np.ndarray]:
        with self._lock:
            items = list(self._buffer)
        if not items:
            return np.empty((0, CHANNEL_COUNT)), np.empty(0)
        timestamps = np.array([it[0] for it in items])
        data       = np.array([it[1] for it in items])
        return data, timestamps

    # ── PPG stubs (LSL inlet carries a single stream; no separate PPG) ────────

    def get_new_ppg_samples(self, from_count: int):
        return np.empty((0, 3)), np.empty(0)

    def get_all_ppg(self):
        return np.empty((0, 3)), np.empty(0)

    def get_ppg_window(self, seconds: float = 10.0):
        return np.empty((0, 3)), np.empty(0)


class LSLOutlet:
    """
    Re-broadcast app data as LSL streams.

    MuseDashboard-EEG     : float32, n_ch × fs — raw samples as they arrive
    MuseDashboard-Markers : string stream — user marker labels
    """

    def __init__(self, name_prefix: str = "MuseDashboard",
                 fs: float = 256.0, n_ch: int = CHANNEL_COUNT,
                 channel_names: list[str] | None = None):
        from pylsl import StreamInfo, StreamOutlet, cf_float32, cf_string

        ch_names = channel_names or ["TP9", "AF7", "AF8", "TP10", "AUX"]
        info = StreamInfo(f"{name_prefix}-EEG", "EEG", n_ch, fs,
                          cf_float32, f"{name_prefix}-eeg-01")
        chns = info.desc().append_child("channels")
        for c in ch_names[:n_ch]:
            chns.append_child("channel").append_child_value("label", c)
        self._eeg_out = StreamOutlet(info)

        minfo = StreamInfo(f"{name_prefix}-Markers", "Markers", 1, 0.0,
                           cf_string, f"{name_prefix}-markers-01")
        self._marker_out = StreamOutlet(minfo)

    def push_eeg(self, data: np.ndarray):
        """Push a chunk of raw samples. data: (n_samples, n_ch) µV."""
        if data is None or data.shape[0] == 0:
            return
        self._eeg_out.push_chunk(data.tolist())

    def push_marker(self, label: str):
        self._marker_out.push_sample([label])
