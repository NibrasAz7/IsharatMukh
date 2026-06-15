# 🧠 Muse S EEG Dashboard

Real-time EEG streaming, signal processing, and mental state monitoring for the **Muse S** headband.

---

## Features

| Feature | Details |
|---|---|
| **Live EEG** | Rolling 4-channel time series (TP9, AF7, AF8, TP10) |
| **Signal cleaning** | Bandpass 0.5–45 Hz + Notch 50/60 Hz (selectable) |
| **EEG Bands** | Delta / Theta / Alpha / Beta / Gamma power bar chart + radar + history |
| **Mental Workload** | Frontal (θ+α)/(α+β) ratio, 0–100 scale, live rolling graph |
| **Mental Fatigue** | Frontal θ/α ratio, 0–100 scale, live rolling graph |
| **Signal quality** | Per-channel colour badges (Good/Fair/Poor) |
| **Save data** | One-click export to **CSV** and **EDF** in `/data/` |
| **Demo mode** | Works without a Muse — synthetic EEG generated automatically |

---

## Quick Start

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

> **Windows note:** If pip fails on `muselsl`, try:
> ```bash
> pip install muselsl --no-build-isolation
> ```

### 2. Turn on your Muse S

Press the power button on your Muse S headband. Make sure your PC Bluetooth is **on**.
No external app (BlueMuse, muselsl CLI) is needed — the dashboard connects directly.

### 3. Run the dashboard

```bash
python app.py
```

Open your browser at **http://localhost:8050**

### 4. Scan and Connect (inside the dashboard)

1. Click **🔍 Scan for Muse** in the sidebar — waits ~10 s for BLE discovery
2. Your headband appears in the **device dropdown** (e.g. `MuseS-1234  (AA:BB:CC:DD:EE:FF)`)
3. Select it and click **▶ Connect**
4. The badge changes: `DEMO` → `● LIVE` when BLE is established

> If no device is found after scanning, try moving the headband closer and scan again.
> If BLE still fails, the dashboard falls back to **DEMO mode** (synthetic EEG so the UI can be explored).

---

## Tabs

| Tab | What you see |
|---|---|
| 📈 Raw EEG | Filtered live EEG waveform. Toggle channels on/off. Adjust window size (2–30 s). |
| 🌊 EEG Bands | Current band powers as grouped bar chart + radar profile + power history over time |
| 🧠 Mental State | Gauges + rolling time series for Mental Workload and Mental Fatigue (0–100%) |

---

## Saving Data

1. Click **⏺ Start Recording** before your session
2. After your session, click **⏹ Stop Recording**
3. Click **💾 Save Data**

Files are saved in the `data/` folder:
- `session_YYYYMMDD_HHMMSS_raw.csv` — raw EEG + timestamps
- `session_YYYYMMDD_HHMMSS_metrics.csv` — workload + fatigue timeseries
- `session_YYYYMMDD_HHMMSS_raw.edf` — EDF format (compatible with MNE, EEGLAB, BrainVision)

---

## Mental State Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| **Mental Workload** | `(θ + α) / (α + β)` frontal | Rises with cognitive demand / task difficulty |
| **Mental Fatigue** | `θ / α` frontal | Rises with sustained mental effort over time |

Both metrics are **normalised 0–100** using a 30-second rolling baseline.
A score > 75 is flagged with a red threshold line.

> **Channels used:** AF7 + AF8 (frontal electrodes) — most sensitive to prefrontal activity.

---

## Settings

| Control | Description |
|---|---|
| Display window | How many seconds of EEG to show (2–30 s) |
| Notch filter | 50 Hz (Europe/UK) or 60 Hz (US/Canada/Japan) |

---

## Project Structure

```
muse_eeg_dashboard/
├── app.py                  # Entry point — run this
├── requirements.txt
├── README.md
├── src/
│   ├── stream_reader.py    # Direct BLE via muselsl+bleak, circular buffer, demo fallback
│   ├── processing.py       # Bandpass/notch filters, band powers, mental metrics
│   ├── data_manager.py     # CSV + EDF file saving
│   └── layout.py           # Dash UI layout (3 tabs + sidebar)
└── data/                   # Auto-created; recordings saved here
```

## How the BLE Connection Works

```
[Muse S headset]
      │  Bluetooth Low Energy
      ▼
[muselsl.Muse (bleak backend)]  ←  pure Python, no external app needed
      │  callback_eeg(data, timestamps)
      ▼
[StreamReader circular buffer]  ←  thread-safe deque, 60 s history
      │  get_window(seconds)
      ▼
[Dash dashboard @ localhost:8050]
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| Scan finds nothing | Move Muse closer; make sure it is powered on (LED blinking); try scanning again |
| Badge stays DEMO | BLE connect failed — check Windows Bluetooth is on; move headband closer |
| BLE errors on Windows | Ensure no other app (e.g. Mind Monitor on phone) is connected to the same headband |
| Data not saving | Ensure `data/` folder is writable; check terminal for errors |
| Notch noise visible | Switch notch filter to match your country’s powerline frequency |
| High "poor" quality | Adjust headband; ensure forehead (AF7/AF8) and ear (TP9/TP10) electrodes have good skin contact |
