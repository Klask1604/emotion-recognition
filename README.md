# Biofizic — server & compute pipeline

Python backend for wrist-based physiological monitoring: MQTT ingestion, HRV analysis, affect heuristics, and Grafana dashboards.

Companion Android app (sensor acquisition): [biofizic-android](https://github.com/YOUR_USER/biofizic) — replace with your repo URL.

## What it does

```
Galaxy Watch (Android app)
  -> MQTT: acquisition/batch @ 1 Hz (IBI + motion stats + cardiac-band energy + HR)
       v
compute-engine     -> HRV 30 s, personal baseline (robust z), signal-quality gate, arousal
mqtt-logger        -> InfluxDB 3
Grafana            -> session dashboards
       v
Watch UI <- biofizic/state/live (1 Hz, hysteresis) + biofizic/state (retained epoch bootstrap)
```

**Production path:** all decisions run in `services/compute_engine.py` → `biofizic/pipeline/PhysiologyPipeline`. There is no separate fusion or ML emotion service in the default stack.

### Outputs (30 s epoch)

| Field | Meaning |
|-------|---------|
| `arousal_10` | 1–10 arousal: `round(1 + 9·Φ(z))` of the personal stress-index z-score once the baseline is locked, else the population Kubios zone |
| `emotion` | Display label (Relaxat … Ridicat) from arousal |
| `emotion_baseline` | Personal-baseline label on the same scale |
| `labels_agree` | Whether the population Kubios zone label matches the personal-baseline label (Cohen's κ over a session) |
| `signal_quality` | Q ∈ [0,1] confidence from the signal-quality gate (IBI artifact rate + cardiac-band motion energy) |
| `alert` | CUSUM sustained-stress alert on the personal z-score |

## Repository layout

```
biofizic/           Core library (signal, HRV, baseline, decision, signal quality, pipeline)
services/           Docker entrypoints (compute_engine, mqtt_logger)
scripts/            Grafana dashboard generator, reports
docker/             InfluxDB + Grafana provisioning
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

2. Build and run (4 services: compute-engine, mqtt-logger, influxdb, grafana):

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

**Watch -> server:** `biofizic/acquisition/batch` (1 Hz schema v2; IBI flushed every 1 s), `biofizic/cmd/calibrate`
**Server -> watch:** `biofizic/state/live` (UI 1 Hz, hysteresis-smoothed), `biofizic/state` (retained epoch decision, doubles as reconnect bootstrap), `biofizic/calibration/status`
**Research/legacy (only when toggled on):** `biofizic/legacy/ppg`, `biofizic/legacy/wesad`, `biofizic/legacy/valence` — never feed VR
**Grafana / Influx (mqtt-logger):** `state`, `state/live`, `state/windows`, `acquisition/batch`, `all_data_live` (unrolled raw PPG/IBI), `legacy/*`

## Production vs. research engines

Production (`biofizic/state` -> VR) is the analytic path: artifact-corrected HRV,
robust per-user log-space baseline, a Kalman smoother on the personal z, a
signal-quality gate, and a CUSUM alert. No trained model artifact is required.

Alongside it, **parallel research engines** can be enabled for the thesis
demonstrations (negative results / comparisons). They publish only on
`biofizic/legacy/*` and **never** feed the VR decision. Each is a build-time
toggle in `biofizic/legacy/toggles.py` (flip + rebuild; default off keeps
production free of scipy/scikit-learn):

| Toggle | What it adds | Needs |
|--------|--------------|-------|
| `ENABLE_RAW_PPG` | parse raw PPG from the watch | `PUBLISH_RAW_PPG` on the watch |
| `ENABLE_PPG_PEAKS` | band-pass + peak detection, PPA, IBI-from-PPG | scipy |
| `ENABLE_WESAD` | WESAD RandomForest P(stress) in parallel | scikit-learn + `python train/train_wesad.py` |
| `ENABLE_VALENCE` | ad-hoc valence heuristic (documented negative result) | raw PPG |

WESAD is a deliberate **domain-shift demonstration** (chest ECG / E4 wrist, not
GW7) — see `docs/THESIS_LIMITATIONS.md`. The earlier WISDM HAR classifier was
retired because its train/serve feature spaces did not match the watch's 1 Hz
aggregated motion stats.

## Development

```bash
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
python services/compute_engine.py --broker localhost --port 1883
```

Human-readable decision logs when running compute-engine:

```
[PHYSIO] activation_level=6/10 (Moderat) | heart_rate=78bpm | rmssd=42ms ...
[QUALITY] motion=still energy=0.02 | artifact_rate=0.01 | quality=0.95 | alert=no | reason=...
[BASELINE] ready=yes | personal_stress_index=10.2 | z_score=+0.45 | labels_match=yes
```

## Tests

Unit tests cover IBI filtering (incl. timestamp-coherent RMSSD), HRV math,
Kubios zone boundaries, the robust log-space baseline z-score, the
signal-quality gate, the CUSUM alert detector, and the live-arousal hysteresis
used to fix the watch UI flicker. Run them with:

```bash
python -m pytest tests/ -v
```

The suite runs in under 2 seconds and has no external dependencies (no MQTT
broker, no InfluxDB needed).

## Validation runbook

A four-layer validation workflow (math unit tests, atomic-sync unit tests,
live diagnostic dashboard, offline replay reproducibility) is documented in
[`docs/VALIDATION.md`](docs/VALIDATION.md). Use that for thesis defence.

Record and replay a session offline with:

```bash
python tools/record_session.py --output session.jsonl --duration 300
python tools/replay_session.py session.jsonl --broker localhost
```

## Grafana dashboards

| UID | Purpose |
|-----|---------|
| `biofizic-live-overview` | Arousal + Kubios label from `biofizic_state_live` (1 Hz) |
| `biofizic-hrv-analysis` | Multi-window RMSSD / stress index (30/60/90 s) |
| `biofizic-baseline-compare` | Kubios vs personal baseline |
| `biofizic-signal-quality` | Motion state, signal quality (Q), IBI artifact rate, cardiac-band motion energy |
| `biofizic-stream-sync` | Atomic-sync diagnostics: anchor delay, IBI count per batch, seq increment |
| `biofizic-window-comparison` | w30/w60/w90 HRV side by side (validation only) |
| `biofizic-session-overview` | Full session timeline |
| `biofizic-all-data-live` | Raw PPG wave + detected peaks + IBI (SDK vs reconstructed) — needs raw-PPG toggles |
| `biofizic-determinist-vs-wesad` | Personal filtered z / arousal vs WESAD P(stress) |
| `biofizic-ppg-failure` | Pulse amplitude collapse under motion (valence-exclusion evidence) |
| `biofizic-valence-demo` | Ad-hoc valence heuristic — documented negative result |

Queries use **`biofizic_state_live`** for live arousal (1 Hz) and **`biofizic_state`** for full 30s epoch decisions.

## License / thesis context

Bachelor thesis project for physiological monitoring in VR/wellness research.
Arousal is derived from the personal stress-index z-score via the normal CDF
(arousal = Φ(z)), with the Kubios stress index (Baevsky 1984; Kubios HRV User
Guide) as the population reference and pre-baseline fallback. HRV reliability is
gated by a signal-quality model — IBI artifact rate (Task Force ESC/NASPE 1996)
and cardiac-band wrist-motion energy — rather than an activity classifier.
Sustained stress is confirmed by a CUSUM change detector (Page 1954). Valence is
**not** estimated. Wrist PPG cannot reliably measure emotional valence.
