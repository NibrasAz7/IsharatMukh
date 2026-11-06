# IsharatMukh — EEG Dashboard (alpha)

**IsharatMukh** (إشارة مخ — "brain signal") is a desktop EEG dashboard built
for the **Muse S** headband, with generic **LSL** support so it works with
any EEG stream in the ecosystem.

## Download

Grab `IsharatMukh-windows-x64-beta-v2.zip` from
[Releases](../../releases), unzip anywhere, and run `IsharatMukh.exe`.
No Python or other installs needed (Windows 10/11, x64).

## Features

- **Live EEG** — rolling/sweep display, TP9, AF7, AF8, TP10 (+AUX)
- **Scan & connect** — BLE device list with RSSI, or resolve any LSL stream
- **Filters** — configurable bandpass + 50/60 Hz notch, live
- **EEG bands** — δ/θ/α/β/γ bars, radar, history
- **Mental metrics** — workload, fatigue, stress, attention, FAA
- **PPG** — HR/HRV from the Muse S sensors
- **Markers/triggers** — user-defined event buttons (keys 1–9)
- **LSL** — read any inlet stream AND publish EEG + markers as outlets
- **Signal quality** — per-channel µV contact-quality readouts
- **Simulate mode** — synthetic EEG, no hardware needed

## Recording

Record → Stop → Save. Output per session in `data/`:

| File | Contents | Compatible with |
|---|---|---|
| `*_data.csv` | timestamp, marker, raw_*, filt_*, sparse metrics | pandas |
| `*_raw.edf` | EDF+ with markers as annotations | **EEGLAB**, MNE |
| `*_raw.fif` | MNE-native format | **MNE** |
| `*_events.tsv` | BIDS events (onset/duration/sample/trial_type) | EEGLAB/MNE |
| `*_meta.json` | device, filters, markers, settings | provenance |
| `*_ppg.csv` | PPG channels @64 Hz | pandas |

## Development

```bash
pip install -r requirements.txt
python app.py
```

Build the distributable exe:

```bash
python -m PyInstaller --noconfirm --clean IsharatMukh.spec
```

## License

GPL-3.0 — see [LICENSE](LICENSE). (PyQt6 binding.)
