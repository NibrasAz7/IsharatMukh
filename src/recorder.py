"""Session-side recording accumulator.

The stream readers only keep a short circular buffer for display.  This class
captures every sample that arrives *while recording* so sessions of any length
are saved in full — decoupled from the display buffer.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np


@dataclass
class SessionRecorder:
    """Accumulates EEG/PPG chunks during a recording session."""

    eeg_chunks: list[np.ndarray] = field(default_factory=list)
    eeg_ts: list[np.ndarray] = field(default_factory=list)
    ppg_chunks: list[np.ndarray] = field(default_factory=list)
    ppg_ts: list[np.ndarray] = field(default_factory=list)
    start: float = 0.0
    stop: float = 0.0

    def start_recording(self) -> None:
        self.eeg_chunks.clear()
        self.eeg_ts.clear()
        self.ppg_chunks.clear()
        self.ppg_ts.clear()
        self.start = time.time()
        self.stop = 0.0

    def append_eeg(self, data: np.ndarray, ts: np.ndarray) -> None:
        if data is not None and data.shape[0] > 0:
            self.eeg_chunks.append(data)
            self.eeg_ts.append(ts)

    def append_ppg(self, data: np.ndarray, ts: np.ndarray) -> None:
        if data is not None and data.shape[0] > 0:
            self.ppg_chunks.append(data)
            self.ppg_ts.append(ts)

    def stop_recording(self) -> None:
        self.stop = time.time()

    def result(self) -> dict:
        """Concatenate chunks into final arrays."""
        eeg_data = (np.vstack(self.eeg_chunks) if self.eeg_chunks
                    else np.empty((0, 5)))
        eeg_ts   = (np.concatenate(self.eeg_ts) if self.eeg_ts
                    else np.empty(0))
        ppg_data = (np.vstack(self.ppg_chunks) if self.ppg_chunks
                    else np.empty((0, 3)))
        ppg_ts   = (np.concatenate(self.ppg_ts) if self.ppg_ts
                    else np.empty(0))
        duration = (self.stop - self.start) if self.stop else 0.0
        return {
            "eeg_data": eeg_data,
            "eeg_ts":   eeg_ts,
            "ppg_data": ppg_data,
            "ppg_ts":   ppg_ts,
            "start":    self.start,
            "stop":     self.stop,
            "duration_s": duration,
        }

    @property
    def n_eeg(self) -> int:
        return sum(c.shape[0] for c in self.eeg_chunks)

    @property
    def n_ppg(self) -> int:
        return sum(c.shape[0] for c in self.ppg_chunks)
