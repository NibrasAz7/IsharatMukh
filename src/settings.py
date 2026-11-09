"""Persistent user settings for IsharatMukh.

Stored as JSON at %APPDATA%/IsharatMukh/settings.json on Windows
(fallback ~/.isharatmukh/settings.json elsewhere).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from src.processing import FilterBank

METRICS = ("workload", "fatigue", "stress", "attention", "faa")

DEFAULT_SETTINGS = {
    "ui_scale": 1.0,                    # 0.9 / 1.0 / 1.15 / 1.3
    "autosave": True,                   # save automatically when recording stops
    "epoch_pad_s": 0.5,                 # padding around threshold-exceeding epochs
    "metric_thresholds": {m: 75.0 for m in METRICS},
    "filter_banks": [],                 # list of bank dicts; empty = defaults
    "active_bank": "Cleaning",
    "marker_defs": ["Start", "Rest", "Task", "End"],
}

UI_SCALE_CHOICES = (0.9, 1.0, 1.15, 1.3)


def _settings_path() -> Path:
    appdata = os.environ.get("APPDATA")
    if appdata:
        return Path(appdata) / "IsharatMukh" / "settings.json"
    return Path.home() / ".isharatmukh" / "settings.json"


def load_settings() -> dict:
    """Load persisted settings merged over defaults."""
    s = json.loads(json.dumps(DEFAULT_SETTINGS))   # deep copy
    path = _settings_path()
    if path.exists():
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
            for k, v in saved.items():
                if k == "metric_thresholds":
                    s["metric_thresholds"].update(
                        {m: float(x) for m, x in v.items()})
                elif k == "filter_banks":
                    s["filter_banks"] = v
                elif k == "marker_defs":
                    s["marker_defs"] = [str(x) for x in v]
                elif k in s:
                    s[k] = v
        except Exception:
            pass
    return s


def save_settings(settings: dict) -> None:
    path = _settings_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(settings, indent=2), encoding="utf-8")
    except Exception as exc:
        print(f"[settings] could not persist: {exc}")


def banks_from_settings(settings: dict) -> tuple[list[dict], str | None]:
    return settings.get("filter_banks", []), settings.get("active_bank")


def banks_to_settings(processor) -> list[dict]:
    return processor.bank_to_dicts()


def bank_dict_valid(d: dict) -> bool:
    try:
        FilterBank.from_dict(d)
        return True
    except Exception:
        return False
