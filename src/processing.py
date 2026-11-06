"""
processing.py
Signal processing pipeline for Muse S EEG data.

Pipeline
--------
raw µV  →  bandpass 0.5–45 Hz  →  notch 50/60 Hz  →  band powers  →  mental metrics
"""

from collections import deque

import numpy as np
from scipy.signal import butter, sosfilt, sosfilt_zi, iirnotch, sosfiltfilt

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
        self.fs              = sample_rate
        self.filter_low      = 0.5
        self.filter_high     = 45.0
        self.notch_freq      = notch_freq
        self.filters_enabled = True
        self._rebuild_filters()

        # Rolling metric history  {metric_name: deque of (time, value)}
        _metrics = ("workload", "fatigue", "stress", "attention", "faa")
        self.metric_history: dict[str, deque] = {
            m: deque(maxlen=METRIC_HISTORY_MAX) for m in _metrics
        }
        # Baseline buffer for normalisation
        self._baseline_buf: deque = deque(maxlen=BASELINE_SECONDS * 2)
        self._baseline_mean: dict = {}
        self._baseline_std:  dict = {}
        # EMA smoothing state
        self._ema_state: dict[str, float] = {}

    # ── Filtering ───────────────────────────────────────────────────────────────

    def _rebuild_filters(self):
        self._bp_sos    = _bandpass_sos(self.filter_low, self.filter_high, self.fs)
        self._notch_sos = _notch_sos(self.notch_freq, self.fs)

    def configure_filters(self, low_hz: float | None = None,
                          high_hz: float | None = None,
                          notch_hz: float | None = None,
                          enabled: bool | None = None):
        """
        Update real-time filter parameters (bandpass low/high cutoff, notch freq,
        master enable).  Callers must reset causal filter states afterwards.
        Raises ValueError on an invalid band.
        """
        low   = self.filter_low  if low_hz   is None else float(low_hz)
        high  = self.filter_high if high_hz  is None else float(high_hz)
        notch = self.notch_freq  if notch_hz is None else float(notch_hz)
        nyq   = self.fs / 2.0
        if not (0.01 <= low < high < nyq):
            raise ValueError(
                f"Invalid band {low:g}–{high:g} Hz (Nyquist {nyq:g} Hz)"
            )
        self.filter_low, self.filter_high, self.notch_freq = low, high, notch
        if enabled is not None:
            self.filters_enabled = bool(enabled)
        self._rebuild_filters()

    def filter_signal(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass + notch filter to a (n_samples, n_channels) array.
        Zero-phase (sosfiltfilt) — use for offline/epoch processing only.
        """
        if data.shape[0] < 10 or not self.filters_enabled:
            return data
        filtered = sosfiltfilt(self._bp_sos,   data, axis=0)
        filtered = sosfiltfilt(self._notch_sos, filtered, axis=0)
        return filtered

    def make_filter_states(self, n_channels: int) -> dict:
        """Create zero initial causal filter states for n_channels."""
        zi_bp    = sosfilt_zi(self._bp_sos)
        zi_notch = sosfilt_zi(self._notch_sos)
        return {
            "bp":    np.zeros((zi_bp.shape[0],    2, n_channels)),
            "notch": np.zeros((zi_notch.shape[0], 2, n_channels)),
        }

    def filter_new(self, data: np.ndarray, states: dict) -> tuple[np.ndarray, dict]:
        """
        Causal online filter — processes only new samples, preserves filter memory.

        Parameters
        ----------
        data   : (n_new_samples, n_channels)
        states : dict with keys "bp" and "notch" (from make_filter_states)

        Returns
        -------
        filtered : (n_new_samples, n_channels)
        states   : updated states (mutated in-place and returned)
        """
        if data.shape[0] == 0 or not self.filters_enabled:
            return data, states
        y, zi_bp    = sosfilt(self._bp_sos,    data, axis=0, zi=states["bp"])
        y, zi_notch = sosfilt(self._notch_sos, y,    axis=0, zi=states["notch"])
        states["bp"]    = zi_bp
        states["notch"] = zi_notch
        return y, states

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

    def compute_metrics_from_powers(self, frontal_powers: dict, timestamp: float,
                                     ch_powers: dict | None = None) -> dict[str, float]:
        """
        Compute all 5 mental metrics from frontal band powers.

        Parameters
        ----------
        frontal_powers : {"theta": float, "alpha": float, "beta": float}
                         Mean power across AF7+AF8.
        timestamp      : Unix time for metric history.
        ch_powers      : Full band_powers dict (needed for per-channel FAA).
                         If None, FAA uses frontal mean (less accurate).

        Returns
        -------
        {"workload", "fatigue", "stress", "attention", "faa"}
        """
        eps     = 1e-9
        theta_f = frontal_powers.get("theta", 0.0)
        alpha_f = frontal_powers.get("alpha", 0.0)
        beta_f  = frontal_powers.get("beta",  0.0)

        # — Existing metrics —
        raw_workload = (theta_f + alpha_f) / (alpha_f + beta_f + eps)
        raw_fatigue  = theta_f / (alpha_f + eps)

        # — Stress: β/α frontal (Alarcao & Fonseca 2017; Pope et al. 1995) —
        raw_stress = beta_f / (alpha_f + eps)

        # — Attention/Engagement: β/(θ+α) frontal (Pope et al. 1995) —
        raw_attention = beta_f / (theta_f + alpha_f + eps)

        # — Frontal Alpha Asymmetry: ln(α_AF8) − ln(α_AF7) (Davidson 1990) —
        if ch_powers is not None:
            alpha_ch = ch_powers.get("alpha", np.zeros(5))
            alpha_af7 = float(alpha_ch[1]) + eps   # AF7 = index 1
            alpha_af8 = float(alpha_ch[2]) + eps   # AF8 = index 2
        else:
            alpha_af7 = alpha_af8 = alpha_f + eps
        raw_faa = float(np.log(alpha_af8) - np.log(alpha_af7))

        raw_vals = {
            "workload": raw_workload, "fatigue": raw_fatigue,
            "stress":   raw_stress,   "attention": raw_attention,
        }
        self._baseline_buf.append(raw_vals)
        self._update_baseline()

        metrics = {}
        for k, v in raw_vals.items():
            metrics[k] = self._ema(k, self._normalise(k, v))
        # FAA: z-score but allow negative (mapped to 0-100 where 50=neutral)
        metrics["faa"] = self._ema("faa", self._normalise_faa(raw_faa))

        for k, v in metrics.items():
            self.metric_history[k].append((timestamp, v))

        return metrics

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
        self._ema_state.clear()

    # ── Signal quality ──────────────────────────────────────────────────────────

    def signal_quality(self, data: np.ndarray) -> list[str]:
        """
        Return a quality label per channel: 'good', 'fair', or 'poor'.
        Based on amplitude RMS (too low = contact issue, too high = artifact).
        """
        return [s["label"] for s in self.channel_stats(data)]

    def channel_stats(self, data: np.ndarray) -> list[dict]:
        """
        Per-channel contact-quality stats for the electrode-health readouts.

        Returns one dict per channel:
            {"rms": µV RMS, "line_noise": µV RMS in ±2 Hz around the notch
             frequency, "label": 'good'|'fair'|'poor'}

        NOTE: the Muse S does not expose true electrode impedance over BLE via
        muselsl — rms/line_noise are a contact-quality proxy, not impedance.
        """
        n_ch = data.shape[1] if data.ndim == 2 else 1
        if data.shape[0] == 0:
            return [{"rms": 0.0, "line_noise": 0.0, "label": "poor"}] * n_ch

        rms = np.sqrt(np.mean(data ** 2, axis=0))

        # Line-noise estimate: integrate Welch PSD in a ±2 Hz band around the
        # configured notch (powerline) frequency.
        from scipy.signal import welch
        line = np.zeros(n_ch)
        if data.shape[0] >= int(self.fs // 2):
            freqs, psd = welch(data, fs=self.fs,
                               nperseg=int(self.fs // 2), axis=0)
            band = np.where((freqs >= self.notch_freq - 2.0)
                            & (freqs <= self.notch_freq + 2.0))[0]
            if len(band):
                df    = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
                line  = np.sqrt(np.sum(psd[band, :], axis=0) * df)

        stats = []
        for i in range(n_ch):
            r = rms[i] if i < len(rms) else 0.0
            if r < 1.0 or r >= 150.0:
                label = "poor"          # no contact / saturated
            elif r < 5.0:
                label = "fair"
            else:
                label = "good"
            stats.append({"rms": float(r), "line_noise": float(line[i]),
                          "label": label})
        return stats

    # ── Private ─────────────────────────────────────────────────────────────────

    def _update_baseline(self):
        """Recompute baseline stats from the rolling buffer."""
        if len(self._baseline_buf) < 10:
            return
        for key in ("workload", "fatigue", "stress", "attention"):
            vals = [d[key] for d in self._baseline_buf if key in d]
            if vals:
                self._baseline_mean[key] = float(np.mean(vals))
                self._baseline_std[key]  = float(np.std(vals)) or 1.0

    def _normalise(self, key: str, raw: float) -> float:
        """Z-score normalise then map to 0–100 (clipped)."""
        if key not in self._baseline_mean:
            scale = {"workload": 3.0, "fatigue": 5.0,
                     "stress": 3.0, "attention": 1.0}.get(key, 2.0)
            return float(np.clip(raw / scale * 100, 0, 100))
        z     = (raw - self._baseline_mean[key]) / (self._baseline_std[key] + 1e-9)
        score = (z + 3) / 6 * 100
        return float(np.clip(score, 0, 100))

    def _normalise_faa(self, raw_faa: float) -> float:
        """Map FAA from [-3σ, +3σ] to [0, 100] where 50=neutral."""
        faa_std = self._baseline_std.get("_faa", 1.0)
        faa_mean = self._baseline_mean.get("_faa", 0.0)
        z = (raw_faa - faa_mean) / (faa_std + 1e-9)
        return float(np.clip((z + 3) / 6 * 100, 0, 100))

    def _ema(self, key: str, new_val: float, alpha: float = 0.3) -> float:
        """Exponential moving average; alpha=0.3 (heavy smoothing)."""
        prev = self._ema_state.get(key, new_val)
        smooth = alpha * new_val + (1.0 - alpha) * prev
        self._ema_state[key] = smooth
        return smooth
