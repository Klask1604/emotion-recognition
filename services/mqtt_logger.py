#!/usr/bin/env python3
"""
MQTT to InfluxDB 3 Core logger.
Subscribes to biofizic topics and writes measurements to local InfluxDB 3 Core.

Usage:
    pip install influxdb-client paho-mqtt
    python mqtt_logger.py
    python mqtt_logger.py --url http://localhost:8181 --database biofizic
"""

import sys
from pathlib import Path

_sys_root = Path(__file__).resolve().parents[1]
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))

import biofizic._bootstrap  # noqa: F401
import argparse
import json
import logging
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import ASYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt_logger")

# ── Configurare topicuri → campuri InfluxDB ────────────────────────────────────

# Numeric fields per topic (Influx measurement = topic with / replaced by _).
FLOAT_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "arousal_10", "arousal_pct", "mean_hr", "rmssd", "stress_index",
        "baseline_si", "z_si", "motion_conf", "rmssd_w30",
        "rmssd_w60", "rmssd_w90", "stress_index_w30",
        "stress_index_w60", "stress_index_w90", "window_sec",
    ],
    "biofizic/state/live": [
        "arousal_10", "arousal_pct", "mean_hr", "rmssd", "stress_index",
        "baseline_si", "z_si", "motion_conf", "window_sec", "ibi_buffer_size",
    ],
    "biofizic/state/windows": [
        "ibi_buffer_size",
        "w30_rmssd", "w30_sdnn", "w30_pnn50", "w30_stress_index", "w30_mean_hr",
        "w30_ibi_count", "w30_covered_seconds",
        "w60_rmssd", "w60_sdnn", "w60_pnn50", "w60_stress_index", "w60_mean_hr",
        "w60_ibi_count", "w60_covered_seconds",
        "w90_rmssd", "w90_sdnn", "w90_pnn50", "w90_stress_index", "w90_mean_hr",
        "w90_ibi_count", "w90_covered_seconds",
    ],
}

# String fields (FlightSQL queryable; tags alone may not appear in schema)
STRING_FIELDS: dict[str, list[str]] = {}

# String fields stored as InfluxDB tags (used for Grafana filtering / coloring).
TAG_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "emotion", "emotion_baseline", "activity_mode", "motion_class", "why",
    ],
    "biofizic/state/live": [
        "emotion", "activity_mode", "motion_class", "data_quality", "window_used",
    ],
    "biofizic/state/windows": [
        "motion_class", "w30_quality", "w60_quality", "w90_quality",
    ],
}

# Boolean fields written as 0/1 floats so Grafana can plot them on axes.
BOOL_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "profile_ready",
        "baseline_ready",
        "labels_agree",
    ],
    "biofizic/state/live": [
        "live",
        "profile_ready",
        "baseline_ready",
    ],
    "biofizic/state/windows": ["baseline_ready"],
}

ALL_TOPICS = list(FLOAT_FIELDS.keys()) + [
    "biofizic/acquisition/batch",
]

# QoS 1 for low-rate epoch decisions so they survive MQTT reconnects.
TOPIC_QOS: dict[str, int] = {
    "biofizic/state": 1,
}

MQTT_KEEPALIVE_SEC = 120

FLOAT_FIELDS.update({
    "biofizic/acquisition/batch": [
        "hr", "skin_temp", "ambient_temp",
        "acc_rms", "acc_p90", "acc_std", "gyro_rms", "gyro_p90", "gyro_std",
    ],
})


def topic_to_measurement(topic: str) -> str:
    return topic.replace("/", "_")


def flatten_windows(payload: dict) -> dict:
    """Flatten nested windows JSON for Influx field/tag mapping."""
    flat: dict = {
        "baseline_ready": payload.get("baseline_ready", False),
        "ibi_buffer_size": payload.get("ibi_buffer_size", 0),
        "motion_class": payload.get("motion_class"),
    }
    for window_label, fields in payload.get("windows", {}).items():
        if not isinstance(fields, dict):
            continue
        for field_name, value in fields.items():
            key = f"{window_label}_{field_name}"
            if field_name == "quality":
                flat[key] = value
            elif value is not None:
                flat[key] = value
    return flat


def flatten_acquisition(payload: dict) -> dict:
    """Flatten nested motion scalars for Influx (arrays omitted)."""
    flat: dict = {
        "ts": payload.get("ts_publish") or payload.get("ts_anchor"),
        "hr": payload.get("hr"),
        "skin_temp": payload.get("skin_temp"),
        "ambient_temp": payload.get("ambient_temp"),
        "display_on": payload.get("display_on"),
        "seq": payload.get("seq"),
    }
    motion = payload.get("motion") or {}
    for key in ("acc_rms", "acc_p90", "acc_std", "gyro_rms", "gyro_p90", "gyro_std"):
        if key in motion:
            flat[key] = motion[key]
    ibi = payload.get("ibi") or {}
    if isinstance(ibi.get("ms"), list):
        flat["ibi_count"] = len(ibi["ms"])
    ppg = payload.get("ppg") or {}
    if isinstance(ppg.get("green"), list):
        flat["ppg_count"] = len(ppg["green"])
    return flat


class MqttInfluxLogger:

    def __init__(self, broker: str, broker_port: int,
                 influx_url: str, influx_database: str):

        self.bucket    = influx_database
        self._msgs_ok  = 0
        self._msgs_err = 0

        # InfluxDB 3 Core: token="ignored", org="ignored", bucket=database name
        self._influx = InfluxDBClient(
            url=influx_url, token="ignored", org="ignored"
        )
        self.write_api = self._influx.write_api(
            write_options=ASYNCHRONOUS,
            success_callback=self._on_write_ok,
            error_callback=self._on_write_err,
        )
        log.info(f"InfluxDB 3 Core: {influx_url}  database={influx_database} (async writes)")

        import uuid
        self.client = mqtt.Client(
            client_id=f"biofizic_influx_logger_{uuid.uuid4().hex[:8]}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect    = self._on_connect
        self.client.on_message    = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=15)
        self.broker       = broker
        self.broker_port  = broker_port

    def _on_write_ok(self, conf, data: str) -> None:
        self._msgs_ok += 1
        if self._msgs_ok % 50 == 0:
            log.info(
                "Wrote %d points to InfluxDB (%d errors)",
                self._msgs_ok, self._msgs_err,
            )

    def _on_write_err(self, conf, data: str, exception: Exception) -> None:
        self._msgs_err += 1
        log.warning("InfluxDB write error: %s", exception)

    def start(self) -> None:
        log.info(f"Conectare MQTT {self.broker}:{self.broker_port}")
        self.client.connect(self.broker, self.broker_port, keepalive=MQTT_KEEPALIVE_SEC)
        try:
            self.client.loop_forever()
        finally:
            self.write_api.close()
            self._influx.close()

    def _on_connect(self, client, userdata, flags, rc, props=None):
        if rc == 0:
            for t in ALL_TOPICS:
                client.subscribe(t, qos=TOPIC_QOS.get(t, 0))
            log.info(f"MQTT conectat — subscris la {len(ALL_TOPICS)} topicuri")
        else:
            log.error(f"MQTT connect failed rc={rc}")

    def _on_disconnect(self, client, userdata, flags, rc, props=None):
        log.warning(f"MQTT deconectat (rc={rc})")

    def _on_message(self, client, userdata, msg):
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            self._msgs_err += 1
            return

        topic = msg.topic
        measurement = topic_to_measurement(topic)

        if topic == "biofizic/state/windows":
            data = flatten_windows(data)
        elif topic == "biofizic/acquisition/batch":
            data = flatten_acquisition(data)

        # Timestamp din payload daca exista, altfel now
        ts_ms = data.get("ts") or data.get("ts_publish") or data.get("ts_anchor")
        if ts_ms and isinstance(ts_ms, (int, float)) and ts_ms > 1e12:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        point = Point(measurement).time(dt)

        written = 0

        # Campuri numerice
        for field in FLOAT_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                try:
                    point.field(field, float(val))
                    written += 1
                except (TypeError, ValueError):
                    pass

        # Tag-uri string
        for field in TAG_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.tag(field, str(val))

        # String fields (queryable in FlightSQL)
        for field in STRING_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.field(field, str(val))
                written += 1

        # Campuri booleane → 0/1
        for field in BOOL_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.field(field, 1.0 if val else 0.0)
                written += 1

        if written == 0:
            return

        try:
            self.write_api.write(bucket=self.bucket, record=point)
        except Exception as e:
            self._msgs_err += 1
            log.warning(f"InfluxDB enqueue error ({topic}): {e}")


def main():
    parser = argparse.ArgumentParser(description="MQTT → InfluxDB 3 Core logger pentru Biofizic")
    parser.add_argument("--broker",   default="paxbespoke.automateflow.ro")
    parser.add_argument("--port",     type=int, default=1883)
    parser.add_argument("--url",      default="http://localhost:8181",
                        help="InfluxDB 3 Core URL (default: http://localhost:8181)")
    parser.add_argument("--database", default="biofizic",
                        help="Numele database-ului InfluxDB 3 (default: biofizic)")
    args = parser.parse_args()

    logger = MqttInfluxLogger(
        broker=args.broker,
        broker_port=args.port,
        influx_url=args.url,
        influx_database=args.database,
    )
    logger.start()


if __name__ == "__main__":
    main()
