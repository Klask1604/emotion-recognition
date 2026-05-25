# Biofizic — physiological monitoring (thesis)

Galaxy Watch 7 acquisition + Python compute server + Grafana.

## Architecture

```
Watch (acquisition only)
  biofizic/ibi/batch, ppg/batch, sensors/batch @ 1 Hz
  biofizic/epoch @ 30s (legacy compat)
       |
       v
  compute-engine (services/compute_engine.py)
  Rolling buffers -> HRV 15/30/60/90s -> baseline -> HAR -> valence -> decision
       |
       +--> biofizic/state, biofizic/state/live, biofizic/combined
       v
mqtt-logger -> InfluxDB -> Grafana
```

## Project layout

```
Licenta/
├── biofizic/
│   ├── constants/       # HRV, Kubios zones, motion caps
│   ├── signal/          # IBI cleaning
│   ├── features/        # HRV metrics
│   ├── windows/         # Multi-window 15/30/60/90s
│   ├── baseline/        # RestBaselineStore (never reset on motion)
│   ├── decision/        # Alert gate, labels, HAR fusion, valence
│   ├── pipeline/        # PhysiologyPipeline
│   └── motion/          # HAR model (WISDM)
├── services/
│   ├── compute_engine.py
│   ├── ppg_processor.py
│   └── mqtt_logger.py
├── train/               # WESAD, WISDM training
├── eval_results/        # Gate reports (JSON)
├── docs/                # Technical report, ML eval, architecture
└── docker/              # Grafana + Influx provisioning
```

Legacy code moved to `Desktop/licenta_archived/`.

## Quick start

```bash
cp .env.example .env
docker compose build
docker compose up -d
python scripts/generate_grafana_dashboards.py
python scripts/generate_ml_evaluation_report.py
```

Grafana: http://localhost:3000 (admin/admin)

Dashboards:
- `biofizic-live-overview`
- `biofizic-hrv-analysis`
- `biofizic-baseline-compare`
- `biofizic-ppg-pipeline`
- `biofizic-motion-har`
- `biofizic-affect-classification`
- `biofizic-session-overview`

## Datasets (local only, not in Git)

- `wesad/` — WESAD for emotion training (~17 GB)
- `datasets/wisdm/extracted/` — WISDM for HAR

Train models:

```bash
python train/train_wisdm_har.py
python train/train_wesad_epoch.py
```

## Watch app

Separate repo: `AndroidStudioProjects/biofizic`

Publishes acquisition batches at 1 Hz. HRV is computed on the server only.

## ML policy

Models deploy only if LOSO gate passes (`eval_results/*.json`). Otherwise deterministic Kubios physiology is used. See `docs/ml_evaluation_report.md`.
