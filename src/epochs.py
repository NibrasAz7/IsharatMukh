"""Epoch extraction — find contiguous segments where mental metrics exceed
their thresholds.

Works on the sparse metric history stored by EEGProcessor.metric_history:
{metric: [(unix_ts, value), ...]} sampled at ~2 s epochs.

Can also run standalone on a saved session CSV:
    python -m src.epochs data/session_X_data.csv [--pad 0.5]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ALL_METRICS = ("workload", "fatigue", "stress", "attention", "faa")
DEFAULT_THRESHOLDS = {m: 75.0 for m in ALL_METRICS}
EPOCH_INTERVAL_S = 2.0        # metrics are emitted ~every 2 s


def _above_any(ts: np.ndarray, vals: np.ndarray,
               thresholds: dict[str, float],
               hist: dict[str, list]) -> np.ndarray:
    """Boolean mask: True at epoch timestamps where ANY metric >= its threshold."""
    above = np.zeros(len(ts), dtype=bool)
    pos = {t: i for i, t in enumerate(ts)}
    for m in ALL_METRICS:
        thr = thresholds.get(m, 75.0)
        for t, v in hist.get(m, []):
            i = pos.get(float(t))
            if i is not None and v >= thr:
                above[i] = True
    return above


def find_epochs(metric_history: dict[str, list],
                thresholds: dict[str, float] | None = None,
                pad_s: float = 0.5,
                t0: float | None = None) -> list[dict]:
    """
    Group above-threshold metric epochs into contiguous segments.

    Parameters
    ----------
    metric_history : {metric: [(ts, value), ...]}  (processor format)
    thresholds     : per-metric thresholds (default 75)
    pad_s          : padding added before/after each segment
    t0             : session start (unix); onsets reported relative to it.
                     If None, relative to first metric timestamp.

    Returns list of dicts:
        onset_s, duration_s, sample(=-1 offline), peak_metric, peak_value,
        mean_{metric} columns
    """
    thresholds = thresholds or DEFAULT_THRESHOLDS

    # Union of all metric timestamps (sorted)
    all_ts = sorted({float(t) for hist in metric_history.values()
                     for t, _ in hist})
    if not all_ts:
        return []
    ts = np.array(all_ts)
    vals = {m: dict(hist) for m, hist in metric_history.items()}

    above = np.array([
        any(vals.get(m, {}).get(float(t), -np.inf) >= thresholds.get(m, 75.0)
            for m in ALL_METRICS)
        for t in ts
    ])

    # Group above-threshold epoch indices; merge groups separated by at most
    # MERGE_GAP_S (≈ one missed metric epoch) into a single epoch.
    above_idx = np.where(above)[0]
    if len(above_idx) == 0:
        return []
    merge_gap = 2.0 * EPOCH_INTERVAL_S
    groups: list[list[int]] = [[int(above_idx[0])]]
    for idx in above_idx[1:]:
        idx = int(idx)
        if ts[idx] - ts[groups[-1][-1]] <= merge_gap:
            groups[-1].append(idx)
        else:
            groups.append([idx])

    offset = t0 if t0 is not None else float(ts[0])
    epochs = []
    for g in groups:
        seg_ts = ts[g]
        onset = float(seg_ts[0]) - pad_s
        # peak metric within segment
        best_m, best_v = None, -np.inf
        for m in ALL_METRICS:
            mv = vals.get(m, {})
            for t in seg_ts:
                v = mv.get(float(t))
                if v is not None and v > best_v:
                    best_v, best_m = v, m
        row = {
            "onset_s": round(onset - offset, 3),
            "duration_s": round(float(seg_ts[-1] - seg_ts[0])
                                + EPOCH_INTERVAL_S + 2 * pad_s, 3),
            "sample": -1,
            "peak_metric": best_m or "",
            "peak_value": round(float(best_v), 2) if best_v > -np.inf else "",
        }
        for m in ALL_METRICS:
            row[f"mean_{m}"] = round(float(np.nanmean([
                vals.get(m, {}).get(float(t), np.nan) for t in seg_ts
            ])), 2)
        epochs.append(row)
    return epochs


def epochs_from_csv(csv_path: str | Path, pad_s: float = 0.5,
                    thresholds: dict[str, float] | None = None) -> Path:
    """Standalone extractor: read a session_X_data.csv, write epochs CSV."""
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    t = df["timestamp"].to_numpy()
    hist = {}
    for m in ALL_METRICS:
        if m in df.columns:
            mask = df[m].notna().to_numpy()
            hist[m] = list(zip(t[mask].tolist(), df.loc[mask, m].tolist()))
    rows = find_epochs(hist, thresholds=thresholds, pad_s=pad_s, t0=t[0])
    if rows:
        # map onset to nearest sample index
        for r in rows:
            r["sample"] = int(np.searchsorted(t, t[0] + r["onset_s"]))
    out = csv_path.with_name(csv_path.name.replace("_data.csv", "_epochs.csv"))
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"[epochs] {len(rows)} epoch(s) → {out}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Extract above-threshold EEG epochs")
    ap.add_argument("csv", help="session_X_data.csv produced by IsharatMukh")
    ap.add_argument("--pad", type=float, default=0.5,
                    help="padding seconds around each epoch (default 0.5)")
    args = ap.parse_args()
    epochs_from_csv(args.csv, pad_s=args.pad)


if __name__ == "__main__":
    main()
