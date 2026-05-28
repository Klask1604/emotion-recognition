# services/

## Purpose
The two runnable processes (Docker). They are the only MQTT clients and the only
InfluxDB writer — the `biofizic/` package is pure compute, these wire it to the world.

## Inputs
- MQTT topics from the watch: `biofizic/acquisition/batch`, `biofizic/cmd/calibrate`.
- (logger) all server-published topics.

## Outputs
- **compute-engine** publishes verdicts: `biofizic/state` (30s, retained QoS1),
  `biofizic/state/live` (1 Hz), `biofizic/state/windows`, `biofizic/live` (aligned 1 Hz),
  `biofizic/calibration/status`, `biofizic/legacy/*`.
- **mqtt-logger** writes every topic to InfluxDB measurements (`biofizic_state`,
  `biofizic_live`, `biofizic_acquisition_batch`, `biofizic_all_data_live`, …).

## Key files
| File | Role |
|---|---|
| `compute_engine.py` | Subscribes, runs `PhysiologyPipeline`, stamps streams with server-now `_anchor_ms`, handles calibrate (collecting→done), publishes verdicts |
| `mqtt_logger.py` | Per-topic field allowlist → InfluxDB points; **dedicated writer thread + queue** (decoupled from MQTT), anchors raw PPG to now, 30s heartbeat |

## Data flow
```
acquisition/batch ─▶ compute_engine ─▶ PhysiologyPipeline ─▶ MQTT (state/live/windows/live)
cmd/calibrate     ─▶ compute_engine.reset_baseline ─▶ calibration/status (collecting→done)

all topics ─▶ mqtt_logger._on_message ─▶ queue ─▶ writer thread ─▶ InfluxDB ─▶ Grafana
```

## Depends on / Used by
- **Depends on:** `biofizic/*` (whole compute package), paho-mqtt, influxdb client.
- **Used by:** the watch (consumes `state`/`calibration/status`), Grafana (reads InfluxDB).
- Reliability: writes run off the MQTT callback thread so a slow write never stalls
  reception (fixed the recurring data gaps); heartbeat exposes `recv/pts/err/queue/dropped`.
