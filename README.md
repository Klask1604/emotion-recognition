# Biofizic server

Real-time affective-state classifier for VR. A Galaxy Watch streams sensor data
over MQTT. This server cleans the signal, computes the state, stores everything in
InfluxDB, and serves Grafana dashboards. The watch app is in a separate repository.

The watch only acquires and sends raw signals. All the heart-rate-variability work,
the personal baseline, the arousal estimate and the valence model run here.

## What the system does

It places the user in the Russell circumplex, on two axes:

- **Arousal** (how activated), from HRV plus a personal baseline. This is the part
  that works well and is validated.
- **Valence** (pleasant vs unpleasant), from PPG pulse-shape morphology. This only
  separates well when the user is activated, so the verdict treats it as reliable
  only above a certain arousal.

The output is a Russell quadrant (calm, activated-positive, activated-negative,
neutral) with the typical emotions for that zone, not a single exact emotion. We do
not claim to read the precise feeling, only the region.

The verdict combines the two axes in a fixed order. Arousal decides calm vs
activated first (it is the reliable part). Only when the user is activated does the
morphology model decide whether it is pleasant or unpleasant.

## Where to start

- This README: what it is and how to run it.
- `ARCHITECTURE.md`: the full system, how arousal and valence are computed, the data
  flow from watch to Grafana.
- `docs/TOPICS.md`: every MQTT topic, what it carries, who publishes and reads it.
- `docs/CODE_MAP.md`: what every Python file does, grouped by folder.
- `docs/research-log/`: the research history (in Romanian), what was tried and why
  most of it did not work. Thesis material.

## How it runs

Two independent processes. They never call each other, they only talk over MQTT.

- `compute-engine` reads the watch telemetry, runs the pipeline, publishes the state.
- `mqtt-logger` listens to every topic and writes it to InfluxDB so Grafana can draw
  the dashboards.

Inside the engine the data flows through fixed stages:

1. Parse the raw MQTT JSON into typed messages (`io/`).
2. Clean the inter-beat-interval series and reject artifacts (`dsp/filters/`).
3. Compute HRV metrics over 30, 60 and 90 second windows (`dsp/hrv/`).
4. Turn features into a verdict (`engine/`): quality gate, personal z-score, fuse the
   HR and HRV channels by quality, Kalman smoothing, CUSUM change gate, arousal mapping.
5. Run the two state models (`research/stari/`): HRV state and PPG morphology, combined
   into the quadrant.
6. Publish the result over MQTT.

## Requirements

- Docker and Docker Compose.
- An MQTT broker reachable from both the watch and the server (Mosquitto on the host
  works).
- Python 3.11 if you want to run a process directly, without Docker.

## Local setup

```bash
cp .env.example .env
```

Edit `.env`:

```
MQTT_BROKER=host.docker.internal
MQTT_PORT=1883
INFLUX_URL=http://influxdb:8181
INFLUX_DATABASE=biofizic
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

If the broker runs on the host (not in Docker), keep
`MQTT_BROKER=host.docker.internal`. The compose file maps that name to the host gateway.

## Run with Docker

```bash
docker compose up -d --build
```

This brings up InfluxDB, a one-shot init container that creates the database, Grafana
with the dashboards provisioned, and the compute engine plus the logger. Then:

- Grafana at `http://localhost:3000` (log in with the `.env` values).
- InfluxDB at `http://localhost:8181`.

Watch the engine receive batches:

```bash
docker compose logs -f compute-engine
```

Stop everything:

```bash
docker compose down
```

## Run without Docker

For development and tests. Create a venv and install:

```bash
python -m venv venv
venv\Scripts\activate          # on Windows
pip install -r requirements.txt
```

Run a process, pointing it at your broker:

```bash
set PYTHONPATH=.
python services/compute_engine.py --broker localhost --port 1883
```

The logger also needs the InfluxDB connection:

```bash
python services/mqtt_logger.py --broker localhost --port 1883 \
    --url http://localhost:8181 --database biofizic
```

## Tests

```bash
pytest
```

They cover the math and the message contracts, no broker or database needed: artifact
correction, HRV metrics against a reference, the robust baseline, channel fusion, the
Kalman smoothing, the CUSUM gate, the arousal mapping.

## Project layout

```
affectus/            Python package, the processing pipeline
  io/                Wire messages + watch-batch to SensorFrame adapter
  contract/          Device capability contract + handshake (for future chest/head)
  dsp/filters/       Inter-beat-interval cleaning and PPG peaks
  dsp/hrv/           HRV metric calculations
  dsp/               The rest of the signal math (baseline, fusion, quality, emotion)
  engine/            Features to verdict (pipeline, decision, channels)
  research/          Parallel research engines, observed-only (see research/README.md)
    stari/           The live state models: HRV state + PPG morphology (the new system)
services/            The MQTT processes
  compute_engine.py  Computes and publishes the verdict
  mqtt_logger.py     Writes every topic to InfluxDB
models/              Trained models (.joblib) + persisted baselines (.npz)
docker/              InfluxDB init script and Grafana provisioning
tests/               Unit tests
docs/                TOPICS.md, research-log/, session reports
docker-compose.yml   The full stack
Dockerfile           The Python image
```
