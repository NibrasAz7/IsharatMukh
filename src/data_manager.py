"""
data_manager.py
Saves EEG recordings to CSV and EDF formats.
"""

import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"


def _ensure_data_dir() -> Path:
    DATA_DIR.mkdir(exist_ok=True)
    return DATA_DIR


def _session_stem() -> str:
    return datetime.now().strftime("session_%Y%m%d_%H%M%S")


# ── CSV ──────────────────────────────────────────────────────────────────────────

def save_csv(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    band_powers_history: list[dict] | None = None,
    metric_history: dict | None = None,
    stem: str | None = None,
) -> str:
    """
    Save raw EEG + optional band powers + metrics to a CSV file.

    Parameters
    ----------
    timestamps          : (n,)            LSL timestamps
    raw_data            : (n, 5)          raw µV per channel
    band_powers_history : list of dicts   one dict per epoch (optional)
    metric_history      : {"workload": [(ts, val), ...], "fatigue": ...}
    stem                : filename stem (auto-generated if None)

    Returns
    -------
    path : str  — absolute path to saved file
    """
    out_dir = _ensure_data_dir()
    stem = stem or _session_stem()
    csv_path = out_dir / f"{stem}_raw.csv"

    ch_names = ["TP9", "AF7", "AF8", "TP10", "AUX"]
    df = pd.DataFrame(raw_data, columns=ch_names)
    df.insert(0, "timestamp", timestamps)
    df.to_csv(csv_path, index=False)
    print(f"[DataManager] Raw EEG saved → {csv_path}")

    # Optionally save metrics separately
    if metric_history:
        metrics_path = out_dir / f"{stem}_metrics.csv"
        rows = []
        workload = dict(metric_history.get("workload", []))
        fatigue  = dict(metric_history.get("fatigue",  []))
        all_ts   = sorted(set(list(workload.keys()) + list(fatigue.keys())))
        for ts in all_ts:
            rows.append({
                "timestamp": ts,
                "workload":  workload.get(ts, np.nan),
                "fatigue":   fatigue.get(ts, np.nan),
            })
        pd.DataFrame(rows).to_csv(metrics_path, index=False)
        print(f"[DataManager] Metrics saved  → {metrics_path}")

    return str(csv_path)


# ── EDF ──────────────────────────────────────────────────────────────────────────

def save_edf(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    sample_rate: float = 256.0,
    stem: str | None = None,
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

    out_dir  = _ensure_data_dir()
    stem     = stem or _session_stem()
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
            "gender":          "",
            "datarecord_duration": 1.0,
            "startdate":       datetime.now(),
        }
        f.setHeader(header)

        ch_headers = []
        for i in range(n_ch):
            ch_headers.append({
                "label":           ch_names[i],
                "dimension":       "uV",
                "sample_rate":     int(sample_rate),
                "physical_max":    1000.0,
                "physical_min":   -1000.0,
                "digital_max":     32767,
                "digital_min":    -32768,
                "prefilter":       "HP:0.5Hz LP:45Hz N:50Hz",
                "transducer":      "EEG",
            })
        f.setSignalHeaders(ch_headers)
        f.writeSamples([raw_data[:, i] for i in range(n_ch)])

    print(f"[DataManager] EDF saved        → {edf_path}")
    return str(edf_path)


# ── Combined save ────────────────────────────────────────────────────────────────

def save_recording(
    timestamps: np.ndarray,
    raw_data:   np.ndarray,
    sample_rate: float = 256.0,
    metric_history: dict | None = None,
) -> dict[str, str]:
    """
    Save both CSV and EDF in one call.

    Returns dict with keys 'csv' and 'edf'.
    """
    stem = _session_stem()
    csv_path = save_csv(timestamps, raw_data, metric_history=metric_history, stem=stem)
    edf_path = save_edf(timestamps, raw_data, sample_rate=sample_rate, stem=stem)
    return {"csv": csv_path, "edf": edf_path}
