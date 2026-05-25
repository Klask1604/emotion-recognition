# Biofizic — server & compute pipeline

Python backend for wrist-based physiological monitoring: MQTT ingestion, HRV analysis, affect heuristics, and Grafana dashboards.

Companion Android app (sensor acquisition): [biofizic-android](https://github.com/YOUR_USER/biofizic) — replace with your repo URL.

## What it does

```
Galaxy Watch (Android app)
  → MQTT: ibi/batch, ppg/batch, sensors/batch @ 1 Hz
       ↓
compute-engine     → HRV 15/30/60/90 s, personal baseline, HAR, arousal + valence
ppg-processor      → PPG DSP, z_pulse_amp (sympathetic proxy)
mqtt-logger        → InfluxDB 3
Grafana            → session dashboards
       ↓
Watch UI ← biofizic/state/live (1 Hz) + biofizic/combined (retained bootstrap ~30s)
```

**Production path:** all decisions run in `services/compute_engine.py` → `biofizic/pipeline/PhysiologyPipeline`. There is no separate fusion or ML emotion service in the default stack.

### Outputs (30 s epoch)

| Field | Meaning |
|-------|---------|
| `arousal_10` | Kubios stress index → 1–10, capped by motion HAR + alert gate |
| `emotion` | Display label (Relaxat … Ridicat) from arousal |
| `valence_10` | Heuristic 1–10 from RMSSD z + PPG pulse amplitude z |
| `affect_quadrant` | calm / activated / tense / depleted |
| `labels_agree` | Kubios zone label vs personal baseline z-label |

## Repository layout

```
biofizic/           Core library (signal, HRV, baseline, decision, motion HAR, pipeline)
services/           Docker entrypoints (compute_engine, ppg_processor, mqtt_logger)
train/              Offline training (WISDM HAR, optional WESAD)
scripts/            Grafana dashboard generator, reports
docker/             InfluxDB + Grafana provisioning
models/             Local .joblib artifacts (not committed — see models/README.md)
data/               Per-user baseline JSON (gitignored)
```

## Requirements

- Docker & Docker Compose
- Python 3.11+ (for local scripts / training)
- An MQTT broker reachable from watch and server
- Samsung Galaxy Watch with Health SDK (see Android repo)

## Quick start

1. Copy environment template and set your broker/database:

```bash
cp .env.example .env
# Edit MQTT_BROKER, MQTT_PORT, INFLUX_DATABASE, Grafana credentials
```

2. Build and run (5 services: compute-engine, ppg-processor, mqtt-logger, influxdb, grafana):

```bash
docker compose build
docker compose up -d
```

3. Regenerate Grafana dashboards after code changes:

```bash
python scripts/generate_grafana_dashboards.py
docker compose restart grafana
```

4. Open Grafana at `http://localhost:3000` (default admin credentials from `.env`).

### Fix duplicate Grafana dashboards

Duplicates usually come from the Grafana data volume keeping old UI copies while file provisioning re-imports.

```bash
docker compose stop grafana
docker volume rm biofizic_grafana-data   # name may be <project>_grafana-data
docker compose up -d grafana
```

Provisioning is configured with `allowUiUpdates: false` so file UIDs stay authoritative.

## MQTT topics (summary)

**Watch → server:** `biofizic/ibi/batch`, `biofizic/ppg/batch`, `biofizic/sensors/batch`, `biofizic/cmd/calibrate`  
**Server → watch:** `biofizic/state/live` (UI 1 Hz), `biofizic/combined` (retained ~30s), `biofizic/calibration/status`  
**Inter-service:** `biofizic/ppg_hrv` (ppg-processor → compute-engine)  
**Grafana / Influx (mqtt-logger):** `state`, `state/live`, `state/windows`, `combined`, `sensors/batch`, `ppg_hrv`, `ppg_pipeline`

## Models & datasets (local only)

Not in Git (see `.gitignore`):

| Path | Purpose |
|------|---------|
| `models/motion_har_wisdm.joblib` | HAR classifier (required) |
| `datasets/wisdm/` | WISDM training data |
| `wesad/` | Optional emotion dataset (~17 GB) |
| `eval_results/` | ML gate reports |

```bash
python train/train_wisdm_har.py
```

## Development

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python services/compute_engine.py --broker localhost --port 1883
```

Human-readable decision logs when running compute-engine:

```
[PHYSIO] activation_level=6/10 (Moderat) | heart_rate=78bpm | rmssd=42ms ...
[AFFECT] valence=6/10 (neutral) | quadrant=calm | z_pulse_amp=+0.2
```

## Grafana dashboards

| UID | Purpose |
|-----|---------|
| `biofizic-live-overview` | Arousal + valence from `biofizic_state_live` (1 Hz) |
| `biofizic-hrv-analysis` | Multi-window RMSSD / stress index |
| `biofizic-baseline-compare` | Kubios vs personal baseline |
| `biofizic-affect-classification` | 2D affect axes + quadrants |
| `biofizic-session-overview` | Full session timeline |
| `biofizic-ppg-pipeline` | Raw PPG reception + motion gate |
| `biofizic-motion-har` | HAR class + acceleration |

Queries use **`biofizic_state_live`** for live affect (1 Hz) and **`biofizic_state`** for full 30s epoch decisions.

## License / thesis context

Bachelor thesis project — physiological monitoring for VR/wellness research. ML emotion classifier (WESAD) is optional and gated; default production uses deterministic Kubios + heuristics.
