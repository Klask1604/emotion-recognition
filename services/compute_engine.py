#!/usr/bin/env python3
"""
Unified compute engine: MQTT ingestion -> physiology pipeline -> state output.

Replaces classifier-v2, solid-engine, and live-context with one service.
Publishes biofizic/state, biofizic/state/live, biofizic/state/windows, and biofizic/combined.
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

from biofizic.config import EPOCH_PUBLISH_INTERVAL_SECONDS, PRIMARY_DECISION_WINDOW_SECONDS
from biofizic.pipeline import PhysiologyPipeline
from biofizic.types import MultiWindowResult, WindowResult
from biofizic.types import IbiBatchMessage, PhysiologyDecision, SensorBatchMessage

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compute_engine")

MOTION_CODE = {"STILL": 0, "SCROLL": 1, "HAND": 2, "WALK": 3}
QUADRANT_CODE = {"calm": 1, "activated": 2, "tense": 3, "depleted": 4}
WINDOWS_PUBLISH_INTERVAL_SECONDS = 5.0
_BEST_WINDOW_SECONDS = {"w30": 30, "w60": 60, "w90": 90}


class ComputeEngineService:
    def __init__(self, broker: str, port: int) -> None:
        self.pipeline = PhysiologyPipeline()
        self._last_live_at = 0.0
        self._last_windows_at = 0.0
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
            ("biofizic/sensors/batch", 0),
            ("biofizic/ppg_hrv", 0),
            ("biofizic/cmd/calibrate", 1),
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        log.info("Compute engine active (batch topics)")

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

        publish_epoch = now - self._last_epoch_at >= EPOCH_PUBLISH_INTERVAL_SECONDS
        if publish_epoch:
            self._last_epoch_at = now

        result = self.pipeline.run(now=now, publish_epoch=publish_epoch)

        if now - self._last_live_at >= 1.0:
            self._last_live_at = now
            self._publish_live(client, result)

        if now - self._last_windows_at >= WINDOWS_PUBLISH_INTERVAL_SECONDS:
            self._last_windows_at = now
            self._publish_windows(client, result)

        if publish_epoch and result.decision:
            self._publish_state(client, result.decision, live=False, result=result)
            self._publish_combined(client, result.decision)

    @staticmethod
    def _window_payload(window: WindowResult) -> dict:
        if window.quality == "unavailable":
            return {
                "rmssd": None,
                "sdnn": None,
                "pnn50": None,
                "stress_index": None,
                "mean_hr": None,
                "quality": window.quality,
                "ibi_count": window.ibi_count,
                "covered_seconds": round(window.covered_seconds, 1),
            }
        return {
            "rmssd": round(window.rmssd_ms, 1),
            "sdnn": round(window.sdnn_ms, 1),
            "pnn50": round(window.pnn50_pct, 1),
            "stress_index": round(window.kubios_stress_index, 3),
            "mean_hr": round(window.mean_hr_bpm, 1),
            "quality": window.quality,
            "ibi_count": window.ibi_count,
            "covered_seconds": round(window.covered_seconds, 1),
        }

    def _decision_payload(
        self,
        decision: PhysiologyDecision,
        *,
        live: bool,
        result: MultiWindowResult | None = None,
    ) -> dict:
        multi = decision.multi_window
        sensor = self.pipeline.state.last_sensor
        window_sec = PRIMARY_DECISION_WINDOW_SECONDS
        if result is not None:
            window_sec = _BEST_WINDOW_SECONDS.get(result.best_window_label, window_sec)
        elif live:
            window_sec = 15

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
            "window_sec": window_sec,
        }
        if result is not None:
            payload["window_used"] = result.best_window_label
            payload["data_quality"] = result.best.quality
            payload["ibi_buffer_size"] = result.ibi_buffer_size
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

    def _publish_live(self, client, result: MultiWindowResult) -> None:
        decision = result.decision
        if decision:
            payload = self._decision_payload(decision, live=True, result=result)
        else:
            payload = {
                "ts": int(time.time() * 1000),
                "live": True,
                "engine": "compute",
                "window_used": result.best_window_label,
                "data_quality": result.best.quality,
                "ibi_buffer_size": result.ibi_buffer_size,
                "motion_class": result.motion_class,
                "baseline_ready": result.baseline_ready,
                "arousal_10": None,
                "valence_10": None,
                "stress_index": None,
                "rmssd": None,
                "mean_hr": None,
                "window_sec": _BEST_WINDOW_SECONDS.get(result.best_window_label, 30),
            }
        client.publish("biofizic/state/live", json.dumps(payload), qos=0)

    def _publish_windows(self, client, result: MultiWindowResult) -> None:
        payload = {
            "ts": int(time.time() * 1000),
            "windows": {
                "w30": self._window_payload(result.w30),
                "w60": self._window_payload(result.w60),
                "w90": self._window_payload(result.w90),
            },
            "motion_class": result.motion_class,
            "baseline_ready": result.baseline_ready,
            "ibi_buffer_size": result.ibi_buffer_size,
        }
        client.publish("biofizic/state/windows", json.dumps(payload), qos=0)

    def _publish_state(
        self,
        client,
        decision: PhysiologyDecision,
        *,
        live: bool,
        result: MultiWindowResult | None = None,
    ) -> None:
        payload = self._decision_payload(decision, live=live, result=result)
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
