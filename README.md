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

### 2. Set up BlueMuse (Windows Bluetooth)

BlueMuse is a free app that handles Bluetooth Low Energy (BLE) pairing on Windows more reliably than the built-in driver.

1. Download from: https://github.com/kowalej/BlueMuse/releases
2. Install and launch **BlueMuse**
3. Turn on your Muse S headband
4. In BlueMuse: click **Start All Streams** — you should see your headband appear

> If you skip BlueMuse, you can also try: `muselsl stream` in a terminal first.

### 3. Run the dashboard

```bash
python app.py
```

Open your browser at **http://localhost:8050**

### 4. Connect

- Click **▶ Connect to Muse** in the sidebar
- The badge will change from `DEMO` → `LIVE` if the stream is found
- If no stream is detected, it falls back to **DEMO mode** (synthetic EEG)

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
│   ├── stream_reader.py    # LSL inlet in background thread → circular buffer
│   ├── processing.py       # Filters, band powers, mental metrics
│   ├── data_manager.py     # CSV + EDF file saving
│   └── layout.py           # Dash UI layout
└── data/                   # Auto-created; recordings saved here
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "No stream found" | Ensure BlueMuse shows your Muse and has "Streams Started" |
| Badge stays DEMO | Check BlueMuse is running before clicking Connect |
| Data not saving | Ensure `data/` folder is writable; check terminal for errors |
| Notch noise visible | Switch notch filter to match your country's powerline frequency |
| High "poor" quality | Adjust headband placement; ensure forehead and ear electrodes have contact |
