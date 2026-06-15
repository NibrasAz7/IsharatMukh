"""
processing.py
Signal processing pipeline for Muse S EEG data.

Pipeline
--------
raw µV  →  bandpass 0.5–45 Hz  →  notch 50/60 Hz  →  band powers  →  mental metrics
"""

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, iirnotch, sosfiltfilt

# ── Band definitions ────────────────────────────────────────────────────────────
BANDS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
    "gamma": (30.0, 45.0),
}

CHANNEL_NAMES   = ["TP9", "AF7", "AF8", "TP10", "AUX"]
FRONTAL_CH_IDX  = [1, 2]          # AF7, AF8 — used for mental metrics
SAMPLE_RATE     = 256.0           # Hz

# Baseline calibration window for normalising the mental metrics
BASELINE_SECONDS    = 30          # collect this many seconds before normalising
METRIC_HISTORY_MAX  = 3600        # keep up to 1 h of metric history


def _bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _notch_sos(freq: float, fs: float, q: float = 30.0):
    b, a = iirnotch(freq / (0.5 * fs), q)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    return tf2sos(b, a)


class EEGProcessor:
    """
    Stateless filter application + stateful metric history.

    Parameters
    ----------
    sample_rate  : Hz (default 256)
    notch_freq   : powerline frequency — 50 (EU) or 60 (US)
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE, notch_freq: float = 50.0):
        self.fs        = sample_rate
        self._bp_sos   = _bandpass_sos(0.5, 45.0, sample_rate)
        self._notch_sos = _notch_sos(notch_freq, sample_rate)

        # Rolling metric history  {metric_name: deque of (time, value)}
        self.metric_history: dict[str, deque] = {
            "workload": deque(maxlen=METRIC_HISTORY_MAX),
            "fatigue":  deque(maxlen=METRIC_HISTORY_MAX),
        }
        # Baseline buffer for normalisation
        self._baseline_buf: deque = deque(maxlen=BASELINE_SECONDS * 2)
        self._baseline_mean: dict = {}
        self._baseline_std:  dict = {}

    # ── Filtering ───────────────────────────────────────────────────────────────

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass + notch filter to a (n_samples, n_channels) array.
        Returns filtered array of same shape.
        """
        if data.shape[0] < 10:
            return data
        filtered = sosfiltfilt(self._bp_sos,   data, axis=0)
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=0)
        return filtered

    # ── Band power ──────────────────────────────────────────────────────────────

    def band_powers(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """
        Compute average band power per channel using Welch's method.

        Parameters
        ----------
        data : (n_samples, n_channels)

        Returns
        -------
        dict: band_name → ndarray of shape (n_channels,), power in µV²
        """
        from scipy.signal import welch
        if data.shape[0] < int(self.fs):
            return {b: np.zeros(data.shape[1]) for b in BANDS}

        freqs, psd = welch(data, fs=self.fs, nperseg=int(self.fs), axis=0)
        # psd shape: (n_freqs, n_channels)

        powers = {}
        for band, (lo, hi) in BANDS.items():
            idx = np.where((freqs >= lo) & (freqs < hi))[0]
            powers[band] = np.mean(psd[idx, :], axis=0) if len(idx) else np.zeros(data.shape[1])
        return powers

    # ── Mental metrics ──────────────────────────────────────────────────────────

    def compute_metrics(self, data: np.ndarray, timestamp: float) -> dict[str, float]:
        """
        Compute Mental Workload and Mental Fatigue from a filtered epoch.

        Uses frontal channels (AF7=idx1, AF8=idx2) only.
        Metrics are normalised 0–100 using a rolling baseline.

        Returns
        -------
        {"workload": float, "fatigue": float, "raw_workload": float, "raw_fatigue": float}
        """
        if data.shape[0] < int(self.fs * 0.5):
            return {"workload": 0.0, "fatigue": 0.0, "raw_workload": 0.0, "raw_fatigue": 0.0}

        frontal = data[:, FRONTAL_CH_IDX]          # (n_samples, 2)
        bp      = self.band_powers(frontal)

        theta_f = float(np.mean(bp["theta"]))
        alpha_f = float(np.mean(bp["alpha"]))
        beta_f  = float(np.mean(bp["beta"]))

        # Guard against division by zero
        eps = 1e-9

        # Mental Workload = (θ + α) / (α + β)   (increases with cognitive load)
        raw_workload = (theta_f + alpha_f) / (alpha_f + beta_f + eps)

        # Mental Fatigue  = θ / α               (increases with fatigue)
        raw_fatigue  = theta_f / (alpha_f + eps)

        # Update baseline
        self._baseline_buf.append({"workload": raw_workload, "fatigue": raw_fatigue})
        self._update_baseline()

        norm_wl = self._normalise("workload", raw_workload)
        norm_fa = self._normalise("fatigue",  raw_fatigue)

        # Append to history
        self.metric_history["workload"].append((timestamp, norm_wl))
        self.metric_history["fatigue"].append((timestamp, norm_fa))

        return {
            "workload":     norm_wl,
            "fatigue":      norm_fa,
            "raw_workload": raw_workload,
            "raw_fatigue":  raw_fatigue,
        }

    def get_metric_history(self, metric: str, last_seconds: float | None = None
                           ) -> tuple[np.ndarray, np.ndarray]:
        """Return (timestamps, values) arrays for a given metric."""
        hist = list(self.metric_history.get(metric, []))
        if not hist:
            return np.array([]), np.array([])
        ts  = np.array([h[0] for h in hist])
        val = np.array([h[1] for h in hist])
        if last_seconds is not None and len(ts) > 0:
            cutoff = ts[-1] - last_seconds
            mask   = ts >= cutoff
            ts, val = ts[mask], val[mask]
        return ts, val

    def reset_baseline(self):
        """Force re-calibration of the normalisation baseline."""
        self._baseline_buf.clear()
        self._baseline_mean.clear()
        self._baseline_std.clear()

    # ── Signal quality ──────────────────────────────────────────────────────────

    def signal_quality(self, data: np.ndarray) -> list[str]:
        """
        Return a quality label per channel: 'good', 'fair', or 'poor'.
        Based on amplitude RMS (too low = contact issue, too high = artifact).
        """
        if data.shape[0] == 0:
            return ["poor"] * data.shape[1] if data.ndim == 2 else ["poor"]
        rms = np.sqrt(np.mean(data ** 2, axis=0))
        labels = []
        for r in rms:
            if r < 1.0:
                labels.append("poor")
            elif r < 5.0:
                labels.append("fair")
            elif r < 150.0:
                labels.append("good")
            else:
                labels.append("poor")   # saturated / artifact
        return labels

    # ── Private ─────────────────────────────────────────────────────────────────

    def _update_baseline(self):
        """Recompute baseline stats from the rolling buffer."""
        if len(self._baseline_buf) < 10:
            return
        for key in ("workload", "fatigue"):
            vals = [d[key] for d in self._baseline_buf]
            self._baseline_mean[key] = float(np.mean(vals))
            self._baseline_std[key]  = float(np.std(vals)) or 1.0

    def _normalise(self, key: str, raw: float) -> float:
        """Z-score normalise then map to 0–100 (clipped)."""
        if key not in self._baseline_mean:
            # No baseline yet — linear stretch assuming typical ranges
            scale = {"workload": 3.0, "fatigue": 5.0}.get(key, 2.0)
            return float(np.clip(raw / scale * 100, 0, 100))
        z     = (raw - self._baseline_mean[key]) / (self._baseline_std[key] + 1e-9)
        score = (z + 3) / 6 * 100          # map [-3σ, +3σ] → [0, 100]
        return float(np.clip(score, 0, 100))
