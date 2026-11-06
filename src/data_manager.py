"""
data_manager.py
Saves EEG recordings — combined session CSV, JSON metadata sidecar,
PPG CSV, and EDF.
"""

import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"

ALL_METRICS = ("workload", "fatigue", "stress", "attention", "faa")
HIGH_THRESHOLD = 75    # must match mental_panel.HIGH_THRESHOLD


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def _session_stem() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


def _resolve_out(out_dir, stem):
    if out_dir is None:
        out_dir = _ensure_data_dir()
    else:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir, stem or _session_stem()


# ── Combined session CSV ─────────────────────────────────────────────────────────

def save_session_csv(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    filtered_data: np.ndarray | None = None,
    metric_history: dict | None = None,
    marker_events: list[tuple[float, str]] | None = None,
    channel_names: list[str] | None = None,
    threshold_flag: bool = True,
    stem: str | None = None,
    out_dir: Path | None = None,
) -> str:
    """
    Save one combined CSV per session.

    Columns
    -------
    timestamp        : Unix timestamp per EEG sample
    marker           : trigger label on the nearest sample row (empty otherwise)
    raw_<CH>         : raw µV per channel
    filt_<CH>        : filtered µV per channel (if filtered_data given)
    workload, fatigue, stress, attention, faa
                     : mental metrics — SPARSE (values only at their ~2 s
                       epoch timestamps, NaN between samples)
    above_threshold  : 1 on epoch rows where any metric >= HIGH_THRESHOLD

    Returns absolute path of the saved file.
    """
    out_dir, stem = _resolve_out(out_dir, stem)
    csv_path = out_dir / f"{stem}_data.csv"

    _default_ch = ["TP9", "AF7", "AF8", "TP10", "AUX"]
    ch_names = channel_names or _default_ch[:raw_data.shape[1]]
    n = raw_data.shape[0]

    df = pd.DataFrame(raw_data, columns=[f"raw_{c}" for c in ch_names])
    df.insert(0, "timestamp", timestamps)

    # Marker column: trigger label on the nearest sample at/after event time
    marker = np.full(n, "", dtype=object)
    for t, label in (marker_events or []):
        idx = int(np.searchsorted(timestamps, t))
        if 0 <= idx < n:
            marker[idx] = f"{marker[idx]};{label}" if marker[idx] else label
    df.insert(1, "marker", marker)

    if filtered_data is not None and filtered_data.shape == raw_data.shape:
        for i, c in enumerate(ch_names):
            df[f"filt_{c}"] = filtered_data[:, i]

    if metric_history:
        epoch_idx: set[int] = set()
        thresh_col = np.full(n, np.nan)
        for m in ALL_METRICS:
            col = np.full(n, np.nan)
            for ts, val in metric_history.get(m, []):
                idx = int(np.searchsorted(timestamps, ts))
                if 0 <= idx < n:
                    col[idx] = val
                    epoch_idx.add(idx)
                    if val >= HIGH_THRESHOLD:
                        thresh_col[idx] = 1.0
            df[m] = col
        if threshold_flag:
            for i in epoch_idx:
                if np.isnan(thresh_col[i]):
                    thresh_col[i] = 0.0
            df["above_threshold"] = thresh_col

    df.to_csv(csv_path, index=False)
    print(f"[DataManager] Session data saved → {csv_path}")
    return str(csv_path)


# ── JSON metadata sidecar ────────────────────────────────────────────────────────

def save_meta_json(meta: dict, stem: str | None = None,
                   out_dir: Path | None = None) -> str:
    """Write <stem>_meta.json — session settings + provenance."""
    out_dir, stem = _resolve_out(out_dir, stem)
    meta_path = out_dir / f"{stem}_meta.json"
    payload = dict(meta)
    payload["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
    print(f"[DataManager] Metadata saved     → {meta_path}")
    return str(meta_path)


def save_ppg_csv(
    timestamps: np.ndarray,
    ppg_data:   np.ndarray,
    stem: str | None = None,
    out_dir: Path | None = None,
) -> str:
    """
    Save raw PPG (Ambient, IR, Red) to a CSV file.

    Returns
    -------
    path : str  — absolute path to saved file
    """
    out_dir, stem = _resolve_out(out_dir, stem)
    csv_path = out_dir / f"{stem}_ppg.csv"

    ch_names = ["Ambient", "IR", "Red"]
    df = pd.DataFrame(ppg_data, columns=ch_names[:ppg_data.shape[1]])
    df.insert(0, "timestamp", timestamps)
    df.to_csv(csv_path, index=False)
    print(f"[DataManager] PPG data saved    → {csv_path}")
    return str(csv_path)


# ── EDF ──────────────────────────────────────────────────────────────────────────

def save_edf(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    sample_rate: float = 256.0,
    stem: str | None = None,
    out_dir: Path | None = None,
    prefilter: str | None = None,
    marker_events: list[tuple[float, str]] | None = None,
) -> str:
    """
    Save raw EEG to EDF format using pyEDFlib.

    Returns
    -------
    path : str  — absolute path to saved file
    """
    try:
        import pyedflib
    except ImportError:
        print("[DataManager] pyedflib not installed — skipping EDF export.")
        return ""

    out_dir, stem = _resolve_out(out_dir, stem)
    edf_path = out_dir / f"{stem}_raw.edf"

    ch_names = ["TP9", "AF7", "AF8", "TP10", "AUX"]
    n_ch     = min(raw_data.shape[1], len(ch_names))

    with pyedflib.EdfWriter(str(edf_path), n_ch, file_type=pyedflib.FILETYPE_EDFPLUS) as f:
        # Write header
        header = {
            "technician":      "",
            "recording_additional": "",
            "patientname":     "MuseS_User",
            "patient_additional": "",
            "patientcode":     "",
            "equipment":       "Muse S",
            "admincode":       "",
            "sex":             "",
            "birthdate":       "",
            "datarecord_duration": 1.0,
            "startdate":       datetime.now(),
        }
        f.setHeader(header)

        ch_headers = []
        for i in range(n_ch):
            ch_headers.append({
                "label":           ch_names[i],
                "dimension":       "uV",
                "sample_frequency": int(sample_rate),
                "physical_max":    1000.0,
                "physical_min":   -1000.0,
                "digital_max":     32767,
                "digital_min":    -32768,
                "prefilter":       prefilter or "none",
                "transducer":      "EEG",
            })
        f.setSignalHeaders(ch_headers)
        f.writeSamples([raw_data[:, i] for i in range(n_ch)])

        # EDF+ annotations → events in EEGLAB (Biosig) and mne.Annotations
        t0 = timestamps[0] if len(timestamps) else 0.0
        for t, label in (marker_events or []):
            onset = float(t) - float(t0)
            if onset >= 0:
                f.writeAnnotation(onset, 0.0, str(label))

    print(f"[DataManager] EDF saved        → {edf_path}")
    return str(edf_path)


# ── BIDS-style events file ───────────────────────────────────────────────────────

def save_events_tsv(
    timestamps: np.ndarray,
    marker_events: list[tuple[float, str]],
    stem: str | None = None,
    out_dir: Path | None = None,
) -> str:
    """
    Write <stem>_events.tsv — BIDS-convention event table readable by
    MNE (pandas/mne.events) and EEGLAB's import-event.

    Columns: onset (s relative to first sample), duration, sample, trial_type.
    """
    out_dir, stem = _resolve_out(out_dir, stem)
    tsv_path = out_dir / f"{stem}_events.tsv"
    t0 = float(timestamps[0]) if len(timestamps) else 0.0
    rows = [
        {"onset": float(t) - t0, "duration": 0.0,
         "sample": int(np.searchsorted(timestamps, t)),
         "trial_type": str(label)}
        for t, label in marker_events
    ]
    pd.DataFrame(rows, columns=["onset", "duration", "sample", "trial_type"]
                 ).to_csv(tsv_path, sep="\t", index=False)
    print(f"[DataManager] Events saved      → {tsv_path}")
    return str(tsv_path)


# ── FIF (MNE-native) ────────────────────────────────────────────────────────────

def save_fif(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    channel_names: list[str],
    marker_events: list[tuple[float, str]] | None,
    sample_rate: float = 256.0,
    stem: str | None = None,
    out_dir: Path | None = None,
) -> str:
    """
    Save as MNE-native <stem>_raw.fif: data in Volts (SI), channels labelled,
    markers stored as mne.Annotations (readable via events_from_annotations).
    """
    try:
        import mne
    except ImportError:
        print("[DataManager] mne not installed — skipping FIF export.")
        return ""

    out_dir, stem = _resolve_out(out_dir, stem)
    fif_path = out_dir / f"{stem}_raw.fif"

    info = mne.create_info(ch_names=list(channel_names), sfreq=sample_rate,
                           ch_types=["eeg"] * len(channel_names))
    raw = mne.io.RawArray(np.asarray(raw_data).T * 1e-6, info)  # µV → V (SI)

    if marker_events:
        t0 = float(timestamps[0]) if len(timestamps) else 0.0
        ann = mne.Annotations(
            onset=[float(t) - t0 for t, _ in marker_events],
            duration=[0.0] * len(marker_events),
            description=[str(l) for _, l in marker_events],
        )
        raw.set_annotations(ann)

    raw.save(str(fif_path), overwrite=True)
    print(f"[DataManager] FIF saved        → {fif_path}")
    return str(fif_path)


# ── Combined save ────────────────────────────────────────────────────────────────

def save_recording(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    filtered_data: np.ndarray | None = None,
    sample_rate: float = 256.0,
    metric_history: dict | None = None,
    marker_events: list[tuple[float, str]] | None = None,
    channel_names: list[str] | None = None,
    meta: dict | None = None,
    ppg_timestamps: np.ndarray | None = None,
    ppg_data:       np.ndarray | None = None,
    save_path: str | Path | None = None,
    formats: set | None = None,
    threshold_flag: bool = True,
) -> dict[str, str]:
    """
    Save a full session: combined CSV, JSON metadata, optional EDF and PPG CSV.

    Returns dict with keys 'csv', 'meta', 'edf', 'ppg_csv' depending on
    formats/data provided.
    """
    if formats is None:
        formats = {"csv", "edf"}
    out_dir = Path(save_path) if save_path else None
    stem    = _session_stem()
    result: dict[str, str] = {}

    if "csv" in formats:
        result["csv"] = save_session_csv(
            timestamps, raw_data,
            filtered_data=filtered_data,
            metric_history=metric_history,
            marker_events=marker_events,
            channel_names=channel_names,
            threshold_flag=threshold_flag,
            stem=stem,
            out_dir=out_dir,
        )
    if "edf" in formats:
        filt = (meta or {}).get("filters", {})
        prefilter = (
            f"HP:{filt.get('low_hz')}Hz LP:{filt.get('high_hz')}Hz "
            f"N:{filt.get('notch_hz')}Hz"
            if filt.get("enabled") else "none"
        )
        result["edf"] = save_edf(
            timestamps, raw_data,
            sample_rate=sample_rate,
            stem=stem,
            out_dir=out_dir,
            prefilter=prefilter,
            marker_events=marker_events,
        )
    if "fif" in formats:
        _ch = channel_names or ["TP9", "AF7", "AF8", "TP10", "AUX"][:raw_data.shape[1]]
        result["fif"] = save_fif(
            timestamps, raw_data,
            channel_names=_ch,
            marker_events=marker_events,
            sample_rate=sample_rate,
            stem=stem,
            out_dir=out_dir,
        )
    if marker_events:
        result["events_tsv"] = save_events_tsv(
            timestamps, marker_events, stem=stem, out_dir=out_dir,
        )
    if ppg_data is not None and ppg_timestamps is not None and ppg_data.shape[0] > 0:
        result["ppg_csv"] = save_ppg_csv(ppg_timestamps, ppg_data, stem=stem,
                                          out_dir=out_dir)
    if meta is not None:
        meta = dict(meta)
        meta["files"] = {k: os.path.basename(v) for k, v in result.items() if v}
        result["meta"] = save_meta_json(meta, stem=stem, out_dir=out_dir)
    return result
