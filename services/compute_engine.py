#!/usr/bin/env python3
"""
Unified compute engine: MQTT ingestion -> physiology pipeline -> state output.

Replaces classifier-v2, solid-engine, and live-context with one service.
Publishes biofizic/state, biofizic/state/live, and biofizic/combined (no fusion service).
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401

import argparse

import paho.mqtt.client as mqtt

from biofizic.constants.hrv import EPOCH_PUBLISH_INTERVAL_SECONDS, PRIMARY_DECISION_WINDOW_SECONDS
from biofizic.pipeline.physiology_pipeline import PhysiologyPipeline
from biofizic.types.samples import IbiBatchMessage, PpgBatchMessage, PhysiologyDecision, SensorBatchMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compute_engine")

MOTION_CODE = {"STILL": 0, "SCROLL": 1, "HAND": 2, "WALK": 3}
QUADRANT_CODE = {"calm": 1, "activated": 2, "tense": 3, "depleted": 4}


class ComputeEngineService:
    def __init__(self, broker: str, port: int) -> None:
        self.pipeline = PhysiologyPipeline()
        self._last_live_at = 0.0
        self._last_epoch_at = 0.0
        self.client = mqtt.Client(
            client_id="biofizic_compute",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker = broker
        self.port = port

    def start(self) -> None:
        self.client.connect(self.broker, self.port, 60)
        self.client.loop_forever()

    def stop(self) -> None:
        self.client.disconnect()

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc != 0:
            return
        topics = [
            ("biofizic/ibi/batch", 0),
            ("biofizic/ppg/batch", 0),
            ("biofizic/sensors/batch", 0),
            ("biofizic/ppg_hrv", 0),
            ("biofizic/epoch", 1),
            ("biofizic/acc/live", 0),
            ("biofizic/hr/live", 0),
            ("biofizic/cmd/calibrate", 1),
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        log.info("Compute engine active (batch + legacy epoch)")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            return

        topic = msg.topic
        now = time.time()

        if topic == "biofizic/cmd/calibrate":
            self.pipeline.reset_baseline()
            self._publish_calibration(client, "Profile baseline reset")
            return

        if topic == "biofizic/ibi/batch":
            batch = IbiBatchMessage(
                timestamp_ms=int(data.get("ts", now * 1000)),
                intervals_ms=[int(x) for x in data.get("ibi_ms", [])],
                timestamps_ms=[int(x) for x in data.get("ibi_ts", data.get("ibi_ts_ms", []))],
            )
            self.pipeline.ingest_ibi_batch(batch)
        elif topic == "biofizic/ppg/batch":
            batch = PpgBatchMessage(
                timestamp_ms=int(data.get("ts", now * 1000)),
                green=[int(x) for x in data.get("green", [])],
                infrared=[int(x) for x in data.get("ir", data.get("infrared", []))],
                sample_timestamps_ms=[int(x) for x in data.get("ts_ms", data.get("ts", []))],
            )
            self.pipeline.ingest_ppg_batch(batch)
        elif topic == "biofizic/ppg_hrv":
            if "z_pulse_amp" in data:
                self.pipeline.ingest_ppg_hrv(float(data["z_pulse_amp"]), now=now)
        elif topic == "biofizic/sensors/batch":
            batch = SensorBatchMessage(
                timestamp_ms=int(data.get("ts", now * 1000)),
                heart_rate_bpm=float(data.get("hr", 0)),
                acceleration_rms=float(data.get("acc_rms", 0)),
                acceleration_p90=float(data.get("acc_p90", 0)),
                acceleration_std=float(data.get("acc_std", 0)),
                gyroscope_rms=float(data.get("gyro_rms", 0)),
                gyroscope_p90=float(data.get("gyro_p90", 0)),
                gyroscope_std=float(data.get("gyro_std", 0)),
                skin_temperature_c=float(data.get("skin_temp", data.get("skin_temp_c", 0))),
                ambient_temperature_c=float(data.get("ambient_temp", data.get("ambient_temp_c", 0))),
                display_on=bool(data.get("displayOn", data.get("display_on", True))),
            )
            self.pipeline.ingest_sensor_batch(batch)
        elif topic == "biofizic/epoch":
            self.pipeline.ingest_legacy_epoch(data)
            self.pipeline.legacy_epoch_metrics(data)
        elif topic == "biofizic/acc/live":
            prev = self.pipeline.state.last_sensor
            batch = SensorBatchMessage(
                timestamp_ms=int(now * 1000),
                heart_rate_bpm=prev.heart_rate_bpm if prev else 0.0,
                acceleration_rms=float(data.get("acc_rms", 0)),
                acceleration_p90=float(data.get("acc_p90", data.get("acc_rms", 0))),
                acceleration_std=prev.acceleration_std if prev else 0.0,
                gyroscope_rms=prev.gyroscope_rms if prev else 0.0,
                gyroscope_p90=prev.gyroscope_p90 if prev else 0.0,
                gyroscope_std=prev.gyroscope_std if prev else 0.0,
                skin_temperature_c=prev.skin_temperature_c if prev else 0.0,
                ambient_temperature_c=prev.ambient_temperature_c if prev else 0.0,
                display_on=prev.display_on if prev else True,
            )
            self.pipeline.ingest_sensor_batch(batch)
        elif topic == "biofizic/hr/live":
            prev = self.pipeline.state.last_sensor
            batch = SensorBatchMessage(
                timestamp_ms=int(now * 1000),
                heart_rate_bpm=float(data.get("hr", 0)),
                acceleration_rms=prev.acceleration_rms if prev else 0.0,
                acceleration_p90=prev.acceleration_p90 if prev else 0.0,
                acceleration_std=prev.acceleration_std if prev else 0.0,
                gyroscope_rms=prev.gyroscope_rms if prev else 0.0,
                gyroscope_p90=prev.gyroscope_p90 if prev else 0.0,
                gyroscope_std=prev.gyroscope_std if prev else 0.0,
                skin_temperature_c=prev.skin_temperature_c if prev else 0.0,
                ambient_temperature_c=prev.ambient_temperature_c if prev else 0.0,
                display_on=prev.display_on if prev else True,
            )
            self.pipeline.ingest_sensor_batch(batch)

        if now - self._last_live_at >= 1.0:
            self._last_live_at = now
            decision = self.pipeline.run_live_tick(now=now)
            if decision:
                self._publish_state(client, decision, live=True)

        if now - self._last_epoch_at >= EPOCH_PUBLISH_INTERVAL_SECONDS:
            self._last_epoch_at = now
            decision = self.pipeline.run_epoch_tick(now=now)
            if decision:
                self._publish_state(client, decision, live=False)
                self._publish_combined(client, decision)

    def _decision_payload(self, decision: PhysiologyDecision, *, live: bool) -> dict:
        multi = decision.multi_window
        sensor = self.pipeline.state.last_sensor
        payload = {
            "ts": int(time.time() * 1000),
            "live": live,
            "engine": "compute",
            "emotion": decision.display_label,
            "emotion_baseline": decision.baseline_label,
            "labels_agree": decision.labels_agree,
            "arousal_10": decision.display_arousal_10,
            "arousal_pct": round(decision.display_arousal_10 * 10.0, 1),
            "valence_10": decision.valence_10,
            "valence_label": decision.valence_label,
            "affect_quadrant": decision.affect_quadrant,
            "affect_quadrant_code": QUADRANT_CODE.get(decision.affect_quadrant, 0),
            "z_pulse_amp": round(decision.z_pulse_amp, 2),
            "confidence": round(decision.motion_confidence, 3),
            "stress_index": round(decision.kubios_stress_index, 3),
            "baseline_si": round(decision.baseline_stress_index, 3),
            "z_si": round(decision.stress_index_z_score, 2),
            "rmssd": round(decision.rmssd_ms, 1),
            "mean_hr": round(decision.mean_heart_rate_bpm, 1),
            "motion_class": decision.motion_class,
            "motion_code": MOTION_CODE.get(decision.motion_class, -1),
            "motion_conf": round(decision.motion_confidence, 3),
            "activity": decision.motion_class,
            "activity_mode": decision.activity_mode,
            "activity_confidence": round(decision.motion_confidence, 3),
            "motion_acc_rms": round(sensor.acceleration_rms, 3) if sensor else 0.0,
            "acc_window_p90": round(sensor.acceleration_p90, 3) if sensor else 0.0,
            "profile_ready": decision.baseline_ready,
            "baseline_ready": decision.baseline_ready,
            "why": decision.decision_reason,
            "window_sec": PRIMARY_DECISION_WINDOW_SECONDS if not live else 15,
        }
        if multi:
            for suffix, metrics in (
                ("w15", multi.window_15_seconds),
                ("w30", multi.window_30_seconds),
                ("w60", multi.window_60_seconds),
                ("w90", multi.window_90_seconds),
            ):
                if metrics and metrics.is_valid:
                    payload[f"rmssd_{suffix}"] = round(metrics.rmssd_ms, 1)
                    payload[f"stress_index_{suffix}"] = round(metrics.kubios_stress_index, 3)
        return payload

    def _publish_state(self, client, decision: PhysiologyDecision, *, live: bool) -> None:
        payload = self._decision_payload(decision, live=live)
        topic = "biofizic/state/live" if live else "biofizic/state"
        client.publish(topic, json.dumps(payload), qos=1 if not live else 0)

    def _publish_combined(self, client, decision: PhysiologyDecision) -> None:
        """Watch-facing combined view (replaces fusion service)."""
        sensor = self.pipeline.state.last_sensor
        payload = {
            "ts": int(time.time() * 1000),
            "engine": "compute",
            "emotion": decision.display_label,
            "emotion_baseline": decision.baseline_label,
            "labels_agree": decision.labels_agree,
            "arousal_10": decision.display_arousal_10,
            "valence_10": decision.valence_10,
            "valence_label": decision.valence_label,
            "affect_quadrant": decision.affect_quadrant,
            "affect_quadrant_code": QUADRANT_CODE.get(decision.affect_quadrant, 0),
            "z_pulse_amp": round(decision.z_pulse_amp, 2),
            "confidence": round(decision.motion_confidence, 3),
            "hr": int(round(decision.mean_heart_rate_bpm)),
            "rmssd": round(decision.rmssd_ms, 1),
            "stress_index": round(decision.kubios_stress_index, 3),
            "acc_rms": round(sensor.acceleration_rms, 3) if sensor else 0.0,
            "motion_class": decision.motion_class,
            "activity_mode": decision.activity_mode,
            "profile_ready": decision.baseline_ready,
        }
        client.publish("biofizic/combined", json.dumps(payload), qos=1, retain=True)

    def _publish_calibration(self, client, message: str) -> None:
        payload = {
            "ts": int(time.time() * 1000),
            "action": "profile",
            "phase": "done",
            "message": message,
            "profile_ready": self.pipeline.baseline.is_ready,
        }
        client.publish("biofizic/calibration/status", json.dumps(payload), qos=1, retain=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    service = ComputeEngineService(args.broker, args.port)

    def shutdown(_a, _b):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    service.start()


if __name__ == "__main__":
    main()
