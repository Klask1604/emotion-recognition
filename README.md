# Biofizic server

This is the server side of Biofizic, a real-time physiological arousal estimator for
virtual reality. A Galaxy Watch streams sensor data over MQTT. This server cleans the
signal, computes a continuous arousal estimate, stores everything in InfluxDB, and
serves dashboards in Grafana. The watch app lives in a separate repository.

The watch only acquires and sends raw signals. All of the heart-rate-variability work,
the personal baseline, and the arousal verdict are computed here.

## How it works

The server runs as two independent processes. Neither one calls the other; they only
talk through MQTT topics.

- `compute-engine` reads the watch telemetry, runs the processing pipeline, and
  publishes the arousal state.
- `mqtt-logger` listens to every topic and writes it into InfluxDB so Grafana can
  draw the dashboards.

Inside the compute engine, data flows through a fixed sequence of stages. Each stage
takes the output of the one before it:

1. Ingestion parses the raw MQTT JSON into typed message objects.
2. The DSP stage cleans the inter-beat-interval series and rejects artifacts.
3. Feature extraction computes HRV metrics (RMSSD, SDNN, stress index, mean HR) over
   30, 60, and 90 second windows.
4. The engine turns the features into a verdict. It gauges signal quality, standardizes
   each feature against the personal baseline with a z-score, fuses the heart-rate and
   variability channels by quality, smooths the result with a scalar Kalman filter, and
   runs a CUSUM change-detection gate.
5. The result is published back over MQTT as a `PhysiologyDecision`.

The Python package is `affectus/`. The folders map to the stages above: `io/` for the
wire messages and the watch→frame adapter, `dsp/filters/` and `dsp/hrv/` (plus the rest
of `dsp/`) for the signal math, `engine/` for the verdict (pipeline, decision, and the
flat channel builder `channels.py`), and `contract/` for the device capability contract
and handshake (so a future chest/head device declares what it carries). `research/` holds
the parallel research engines that never feed the production arousal verdict — they are
observed-only on `biofizic/legacy/*` and `affectus/research/README.md` explains each one.

## Requirements

- Docker and Docker Compose, for the normal way to run the stack.
- An MQTT broker reachable from both the watch and the server. Mosquitto on the host
  works fine.
- Python 3.11 if you want to run a process directly, without Docker.

## Local setup

Clone the repository and copy the environment template:

```bash
cp .env.example .env
```

Edit `.env` and fill in your own values:

```
MQTT_BROKER=host.docker.internal
MQTT_PORT=1883
INFLUX_URL=http://influxdb:8181
INFLUX_DATABASE=biofizic
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=admin
```

If the broker runs natively on the host (not in Docker), set
`MQTT_BROKER=host.docker.internal` so the containers can reach it. The compose file
already maps that name to the host gateway.

## Run with Docker

Start the whole stack:

```bash
docker compose up -d --build
```

This brings up four things: InfluxDB, a one-shot init container that creates the
database, Grafana with the dashboards already provisioned, and the compute engine and
logger. Once it is up:

- Grafana is at `http://localhost:3000` (log in with the values from `.env`).
- InfluxDB is at `http://localhost:8181`.

Watch the logs to confirm the engine is receiving batches:

```bash
docker compose logs -f compute-engine
```

Stop everything:

```bash
docker compose down
```

### Optional services

Two extra services exist behind the `test` profile and do not start by default.

The cardiac comparator rolls the production DSP over three watch signal sources and
republishes the derived HR and RMSSD for the Grafana comparator board:

```bash
docker compose --profile test up -d test-engine
```

The state API is a manual publisher for VR testing. It lets you drive scene transitions
over HTTP without wearing the watch, by publishing the same `biofizic/state` schema that
Unity reads:

```bash
docker compose up -d state-api
```

It then listens on `http://localhost:8200`.

## Run without Docker

Useful for development and for running the tests. Create a virtual environment and
install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate          # on Windows
pip install -r requirements.txt
```

Run a single process directly, pointing it at your broker:

```bash
set PYTHONPATH=.
python services/compute_engine.py --broker localhost --port 1883
```

The logger needs the InfluxDB connection as well:

```bash
python services/mqtt_logger.py --broker localhost --port 1883 \
    --url http://localhost:8181 --database biofizic
```

## Tests

The unit tests cover the math and the message contracts. They run without a broker or
a database:

```bash
pytest
```

They check artifact correction, the HRV metrics against a reference oracle, the robust
baseline, the quality-weighted channel fusion, the Kalman smoothing, the CUSUM gate, and
the arousal mapping.

## MQTT topics

From the watch to the server:

| Topic | Rate | Content |
|---|---|---|
| `biofizic/acquisition/batch` | 1 Hz | IBI, accelerometer and gyroscope stats, skin temperature, a shared timestamp anchor, and sync diagnostics |
| `biofizic/cmd/calibrate` | on demand | Recalibration request with the reported arousal |

From the server to the watch and the dashboards:

| Topic | Rate | Content |
|---|---|---|
| `biofizic/state` | 30 s | The committed epoch verdict, retained, QoS 1 |
| `biofizic/state/live` | 1 Hz | The live verdict for smooth display |
| `biofizic/state/windows` | 30 s | HRV features over the 30, 60, and 90 second windows |
| `biofizic/live` | 1 Hz | The aligned stream, every field on the shared timestamp anchor |
| `biofizic/calibration/status` | on event | Calibration phase (collecting or done) |
| `biofizic/legacy/{ppg,wesad,valence}` | varies | Research engines, never used for the production verdict |

## Project layout

```
affectus/            Python package with the processing pipeline
  io/                Wire messages + the watch-batch -> SensorFrame adapter
  contract/          Device capability contract + handshake (capabilities, frame, handshake)
  dsp/filters/       Inter-beat-interval cleaning and PPG peaks
  dsp/hrv/           HRV metric calculations
  dsp/               The rest of the signal math (baseline, fusion, quality, ...)
  engine/            Features to verdict (pipeline, decision, flat channel builder)
  research/          Parallel research engines, observed-only (see research/README.md)
services/            The MQTT processes
  compute_engine.py  Computes and publishes the verdict
  mqtt_logger.py     Writes every topic to InfluxDB
  state_api.py       Manual state publisher for VR testing
  test_engine.py     Cardiac comparator
scripts/             Dashboard generation and the WESAD comparison report
train/               Trains the WESAD RandomForest model used by the research engine
docker/              InfluxDB init script and Grafana provisioning
tests/               Unit tests
docker-compose.yml   The full stack
Dockerfile           The Python image
```

## Related documentation

- `ARHITECTURA.md` for the architecture in more detail.
- `docs/` for the narrative explanation and the session reports.
- The Android watch app has its own README in its repository.
