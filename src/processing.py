"""
processing.py
Signal processing pipeline for Muse S EEG data.

Pipeline
--------
raw µV  →  bandpass 0.5–45 Hz  →  notch 50/60 Hz  →  band powers  →  mental metrics
"""

from collections import deque
from dataclasses import dataclass, field

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

# ── Filter banks ────────────────────────────────────────────────────────────────

STAGE_KINDS = ("bandpass", "notch", "highpass", "lowpass")

# Quick-add band buttons in the filter dialog
BAND_PRESETS = {
    "Delta":  (0.5, 4.0),
    "Theta":  (4.0, 8.0),
    "Alpha":  (8.0, 13.0),
    "Beta":   (13.0, 30.0),
    "Gamma":  (30.0, 45.0),
}

# Named one-click bank presets
FILTER_PRESETS = {
    "Cleaning (0.5–45 Hz + notch)": [
        {"kind": "bandpass", "low": 0.5, "high": 45.0, "enabled": True},
        {"kind": "notch",    "freq": 50.0, "enabled": True},
    ],
    "Delta 0.5–4 Hz":   [{"kind": "bandpass", "low": 0.5, "high": 4.0,  "enabled": True}],
    "Theta 4–8 Hz":     [{"kind": "bandpass", "low": 4.0, "high": 8.0,  "enabled": True}],
    "Alpha 8–13 Hz":    [{"kind": "bandpass", "low": 8.0, "high": 13.0, "enabled": True}],
    "Beta 13–30 Hz":    [{"kind": "bandpass", "low": 13.0, "high": 30.0, "enabled": True}],
    "Gamma 30–45 Hz":   [{"kind": "bandpass", "low": 30.0, "high": 45.0, "enabled": True}],
    "EMG clean (>20 Hz)": [
        {"kind": "highpass", "low": 20.0, "enabled": True},
        {"kind": "notch",    "freq": 50.0, "enabled": True},
    ],
    "Wideband 1–40 Hz": [
        {"kind": "bandpass", "low": 1.0, "high": 40.0, "enabled": True},
    ],
}


@dataclass
class FilterStage:
    """One filter stage inside a bank."""
    kind: str = "bandpass"          # bandpass | notch | highpass | lowpass
    low: float = 0.5                # bandpass/highpass lower cutoff (Hz)
    high: float = 45.0              # bandpass/lowpass upper cutoff (Hz)
    freq: float = 50.0              # notch centre frequency (Hz)
    enabled: bool = True

    def describe(self) -> str:
        mark = "✓" if self.enabled else "·"
        if self.kind == "bandpass":
            return f"{mark} Bandpass {self.low:g}–{self.high:g} Hz"
        if self.kind == "notch":
            return f"{mark} Notch {self.freq:g} Hz"
        if self.kind == "highpass":
            return f"{mark} Highpass {self.low:g} Hz"
        return f"{mark} Lowpass {self.high:g} Hz"

    def to_dict(self) -> dict:
        return {"kind": self.kind, "low": self.low, "high": self.high,
                "freq": self.freq, "enabled": self.enabled}

    @staticmethod
    def from_dict(d: dict) -> "FilterStage":
        return FilterStage(
            kind=str(d.get("kind", "bandpass")),
            low=float(d.get("low", 0.5)),
            high=float(d.get("high", 45.0)),
            freq=float(d.get("freq", 50.0)),
            enabled=bool(d.get("enabled", True)),
        )


@dataclass
class FilterBank:
    """A named, ordered chain of filter stages."""
    name: str
    stages: list = field(default_factory=list)

    def enabled_stages(self) -> list:
        return [s for s in self.stages if s.enabled]

    def to_dict(self) -> dict:
        return {"name": self.name,
                "stages": [s.to_dict() for s in self.stages]}

    @staticmethod
    def from_dict(d: dict) -> "FilterBank":
        return FilterBank(
            name=str(d.get("name", "bank")),
            stages=[FilterStage.from_dict(s) for s in d.get("stages", [])],
        )


def _bandpass_sos(low: float, high: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    return butter(order, [low / nyq, high / nyq], btype="band", output="sos")


def _notch_sos(freq: float, fs: float, q: float = 30.0):
    b, a = iirnotch(freq / (0.5 * fs), q)
    # Convert to SOS for numerical stability
    from scipy.signal import tf2sos
    return tf2sos(b, a)


def _stage_sos(stage: FilterStage, fs: float):
    """Compile one FilterStage into an SOS array. Raises ValueError if invalid."""
    nyq = 0.5 * fs
    if stage.kind == "bandpass":
        if not (0.01 <= stage.low < stage.high < nyq):
            raise ValueError(
                f"Invalid bandpass {stage.low:g}–{stage.high:g} Hz (Nyquist {nyq:g})")
        return butter(4, [stage.low / nyq, stage.high / nyq], btype="band",
                      output="sos")
    if stage.kind == "highpass":
        if not (0.01 <= stage.low < nyq):
            raise ValueError(f"Invalid highpass {stage.low:g} Hz (Nyquist {nyq:g})")
        return butter(4, stage.low / nyq, btype="high", output="sos")
    if stage.kind == "lowpass":
        if not (0.01 <= stage.high < nyq):
            raise ValueError(f"Invalid lowpass {stage.high:g} Hz (Nyquist {nyq:g})")
        return butter(4, stage.high / nyq, btype="low", output="sos")
    if stage.kind == "notch":
        if not (0.01 <= stage.freq < nyq):
            raise ValueError(f"Invalid notch {stage.freq:g} Hz (Nyquist {nyq:g})")
        return _notch_sos(stage.freq, fs)
    raise ValueError(f"Unknown stage kind: {stage.kind}")


class EEGProcessor:
    """
    Stateless filter application + stateful metric history.

    Parameters
    ----------
    sample_rate  : Hz (default 256)
    notch_freq   : powerline frequency — 50 (EU) or 60 (US)
    """

    def __init__(self, sample_rate: float = SAMPLE_RATE, notch_freq: float = 50.0):
        self.fs          = sample_rate
        self.notch_freq  = notch_freq      # line-noise reference for channel_stats
        self._banks: dict[str, FilterBank] = {}
        self.active_bank = "Cleaning"
        self._bank_sos: dict[str, list] = {}

        # Default bank preserves the historical 0.5–45 Hz BP + notch behaviour
        self._banks[self.active_bank] = FilterBank(
            self.active_bank,
            stages=[
                FilterStage("bandpass", low=0.5, high=45.0),
                FilterStage("notch", freq=self.notch_freq),
            ],
        )
        self._rebuild_bank(self.active_bank)

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

    # ── Filter banks ─────────────────────────────────────────────────────────────

    @property
    def banks(self) -> dict[str, FilterBank]:
        return self._banks

    @property
    def filters_enabled(self) -> bool:
        return bool(self._banks[self.active_bank].enabled_stages())

    # Legacy scalar accessors (first bandpass / notch of active bank) so older
    # callers and meta dicts keep working.
    @property
    def filter_low(self) -> float:
        for s in self._banks[self.active_bank].stages:
            if s.kind == "bandpass":
                return s.low
        return 0.5

    @property
    def filter_high(self) -> float:
        for s in self._banks[self.active_bank].stages:
            if s.kind == "bandpass":
                return s.high
        return 45.0

    def _rebuild_bank(self, name: str):
        """Compile SOS for every enabled stage of a bank; sync notch_freq."""
        bank = self._banks[name]
        self._bank_sos[name] = [
            _stage_sos(s, self.fs) for s in bank.enabled_stages()
        ]
        if name == self.active_bank:
            for s in bank.stages:
                if s.kind == "notch" and s.enabled:
                    self.notch_freq = s.freq
                    break

    def bank_names(self) -> list[str]:
        return list(self._banks.keys())

    def set_active_bank(self, name: str):
        if name not in self._banks:
            raise KeyError(f"Unknown filter bank: {name}")
        self.active_bank = name
        self._rebuild_bank(name)

    def add_bank(self, name: str, stages: list | None = None) -> FilterBank:
        if name in self._banks:
            raise ValueError(f"Bank '{name}' already exists")
        bank = FilterBank(name, stages=list(stages) if stages else [])
        self._banks[name] = bank
        self._rebuild_bank(name)
        return bank

    def duplicate_bank(self, name: str, new_name: str | None = None) -> FilterBank:
        src = self._banks[name]
        new_name = new_name or f"{name} copy"
        bank = FilterBank(new_name,
                          stages=[FilterStage(**vars(s)) for s in src.stages])
        self._banks[new_name] = bank
        self._rebuild_bank(new_name)
        return bank

    def update_bank(self, name: str, stages: list):
        self._banks[name].stages = list(stages)
        self._rebuild_bank(name)

    def remove_bank(self, name: str):
        if name == self.active_bank:
            raise ValueError("Cannot remove the active bank")
        del self._banks[name]
        self._bank_sos.pop(name, None)

    def bank_to_dicts(self) -> list[dict]:
        return [b.to_dict() for b in self._banks.values()]

    def load_banks(self, bank_dicts: list[dict], active: str | None):
        """Replace all banks from serialized dicts (settings restore)."""
        self._banks = {}
        self._bank_sos = {}
        for d in bank_dicts:
            bank = FilterBank.from_dict(d)
            self._banks[bank.name] = bank
        if not self._banks:
            self._banks["Cleaning"] = FilterBank(
                "Cleaning",
                [FilterStage("bandpass", 0.5, 45.0),
                 FilterStage("notch", freq=self.notch_freq)])
        self.active_bank = active if active in self._banks else next(iter(self._banks))
        for name in self._banks:
            self._rebuild_bank(name)

    def configure_filters(self, low_hz: float | None = None,
                          high_hz: float | None = None,
                          notch_hz: float | None = None,
                          enabled: bool | None = None):
        """
        Legacy convenience: rewrite the active bank as bandpass+notch.
        Prefer editing banks via the filter dialog.
        """
        bank = self._banks[self.active_bank]
        low   = self.filter_low  if low_hz   is None else float(low_hz)
        high  = self.filter_high if high_hz  is None else float(high_hz)
        notch = self.notch_freq  if notch_hz is None else float(notch_hz)
        bank.stages = [
            FilterStage("bandpass", low=low, high=high, enabled=bool(enabled)),
            FilterStage("notch", freq=notch, enabled=bool(enabled)),
        ]
        self._rebuild_bank(self.active_bank)

    # ── Filtering ───────────────────────────────────────────────────────────────

    @staticmethod
    def _apply_stages(data: np.ndarray, sos_list: list,
                      causal: bool = False, states: dict | None = None):
        """Run an (n,ch) array through a list of SOS filters."""
        y = data
        if causal:
            for i, sos in enumerate(sos_list):
                y, zi = sosfilt(sos, y, axis=0, zi=states[str(i)])
                states[str(i)] = zi
            return y
        for sos in sos_list:
            y = sosfiltfilt(sos, y, axis=0)
        return y

    def filter_signal(self, data: np.ndarray,
                      bank: str | None = None) -> np.ndarray:
        """
        Zero-phase (sosfiltfilt) filtering — offline/epoch processing only.
        `bank` selects which bank to apply; default = active bank.
        """
        name = bank or self.active_bank
        sos_list = self._bank_sos.get(name, [])
        if data.shape[0] < 10 or not sos_list:
            return data
        return self._apply_stages(data, sos_list)

    def filter_all_banks(self, data: np.ndarray) -> dict[str, np.ndarray]:
        """Return {bank_name: filtered_data} for every bank with ≥1 stage."""
        out = {}
        for name, bank in self._banks.items():
            if bank.enabled_stages():
                out[name] = self.filter_signal(data, bank=name)
        return out

    def make_filter_states(self, n_channels: int) -> dict:
        """Zero causal states for the ACTIVE bank's enabled stages."""
        states = {}
        for i, sos in enumerate(self._bank_sos.get(self.active_bank, [])):
            zi = sosfilt_zi(sos)
            states[str(i)] = np.zeros((zi.shape[0], 2, n_channels))
        return states

    def filter_new(self, data: np.ndarray, states: dict
                   ) -> tuple[np.ndarray, dict]:
        """
        Causal online filter on the ACTIVE bank — preserves stage memory.
        `states` keys are stage indices as strings (from make_filter_states).
        """
        sos_list = self._bank_sos.get(self.active_bank, [])
        if data.shape[0] == 0 or not sos_list:
            return data, states
        return self._apply_stages(data, sos_list, causal=True, states=states), states

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
