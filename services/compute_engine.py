#!/usr/bin/env python3
"""
Unified compute engine: MQTT ingestion -> physiology pipeline -> state output.

Publishes three MQTT topics:
  - biofizic/state         epoch decision (1 per 30 s), retained for bootstrap
  - biofizic/state/live    smoothed live arousal (1 Hz) for the watch UI
  - biofizic/state/windows w30/w60/w90 HRV side-by-side (every 5 s) for the
                           thesis window-comparison dashboard, validation only
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

from biofizic.config import (
    EPOCH_PUBLISH_INTERVAL_SECONDS,
    LIVE_AROUSAL_HYSTERESIS_TICKS,
    PRIMARY_DECISION_WINDOW_SECONDS,
    WINDOWS_PUBLISH_INTERVAL_SECONDS,
)
from biofizic.decision.arousal_mapper import arousal_scale_10_to_label
from biofizic.pipeline import PhysiologyPipeline
from biofizic.types import AcquisitionBatchMessage, MultiWindowResult, WindowResult
from biofizic.types import PhysiologyDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compute_engine")

MOTION_CODE = {"STILL": 0, "SCROLL": 1, "HAND": 2, "WALK": 3}


def _parse_acquisition(data: dict) -> AcquisitionBatchMessage | None:
    schema = int(data.get("schema", 0))
    if schema < 2:
        return None
    ts_publish = int(data.get("ts_publish") or data.get("ts") or 0)
    if ts_publish <= 0:
        return None
    motion = data.get("motion") or {}
    ibi = data.get("ibi") or {}
    ppg = data.get("ppg") or {}
    ibi_ms = [int(x) for x in (ibi.get("ms") or data.get("ibi_ms") or [])]
    ibi_ts = [int(x) for x in (ibi.get("ts") or data.get("ibi_ts") or [])]
    ppg_ts = [int(x) for x in (ppg.get("ts_ms") or [])]
    ppg_green = [int(x) for x in (ppg.get("green") or [])]
    ppg_ir = [int(x) for x in (ppg.get("ir") or [])]
    anchor_candidates = [ts_publish]
    if ibi_ts:
        anchor_candidates.append(max(ibi_ts))
    if ppg_ts:
        anchor_candidates.append(max(ppg_ts))
    skin_ts = int(data.get("skin_temp_ts") or 0)
    if skin_ts > 0:
        anchor_candidates.append(skin_ts)
    ts_anchor = int(data.get("ts_anchor") or max(anchor_candidates))
    return AcquisitionBatchMessage(
        timestamp_publish_ms=ts_publish,
        timestamp_anchor_ms=ts_anchor,
        sequence=int(data.get("seq") or 0),
        heart_rate_bpm=float(data.get("hr") or 0),
        display_on=bool(data.get("display_on", data.get("displayOn", True))),
        skin_temperature_c=float(data.get("skin_temp") or data.get("skin_temp_c") or 0),
        ambient_temperature_c=float(data.get("ambient_temp") or data.get("ambient_temp_c") or 0),
        skin_temperature_ts_ms=skin_ts,
        acceleration_rms=float(motion.get("acc_rms") or data.get("acc_rms") or 0),
        acceleration_p90=float(motion.get("acc_p90") or data.get("acc_p90") or 0),
        acceleration_std=float(motion.get("acc_std") or data.get("acc_std") or 0),
        gyroscope_rms=float(motion.get("gyro_rms") or data.get("gyro_rms") or 0),
        gyroscope_p90=float(motion.get("gyro_p90") or data.get("gyro_p90") or 0),
        gyroscope_std=float(motion.get("gyro_std") or data.get("gyro_std") or 0),
        motion_window_ms=int(motion.get("window_ms") or 1000),
        ibi_intervals_ms=ibi_ms,
        ibi_timestamps_ms=ibi_ts,
        ibi_timestamp_source=str(ibi.get("source") or "reconstructed"),
        ppg_green=ppg_green,
        ppg_infrared=ppg_ir,
        ppg_timestamps_ms=ppg_ts,
    )


class ComputeEngineService:
    def __init__(self, broker: str, port: int) -> None:
        self.pipeline = PhysiologyPipeline()
        self._last_live_at = 0.0
        self._last_windows_at = 0.0
        self._last_epoch_at = 0.0
        # Streak-based hysteresis for the live arousal integer. We only flip
        # the displayed value when a different raw value has appeared for
        # LIVE_AROUSAL_HYSTERESIS_TICKS consecutive live ticks. This kills
        # the 1-tick alternation that median smoothing would still let through.
        self._live_displayed_a10: int | None = None
        self._live_candidate_a10: int | None = None
        self._live_candidate_streak: int = 0
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
            ("biofizic/acquisition/batch", 0),
            ("biofizic/cmd/calibrate", 1),
        ]
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        log.info("Compute engine active (acquisition/batch v2)")

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

        if topic == "biofizic/acquisition/batch":
            batch = _parse_acquisition(data)
            if batch is None:
                return
            self.pipeline.ingest_acquisition(batch)
            self._run_and_publish(
                client,
                now=now,
                end_timestamp_ms=batch.timestamp_anchor_ms,
            )
            return

        return

    def _run_and_publish(
        self,
        client,
        *,
        now: float,
        end_timestamp_ms: int | None = None,
    ) -> None:
        publish_epoch = now - self._last_epoch_at >= EPOCH_PUBLISH_INTERVAL_SECONDS
        if publish_epoch:
            self._last_epoch_at = now

        result = self.pipeline.run(
            now=now,
            end_timestamp_ms=end_timestamp_ms,
            publish_epoch=publish_epoch,
        )

        if now - self._last_live_at >= 1.0:
            self._last_live_at = now
            self._publish_live(client, result)

        if now - self._last_windows_at >= WINDOWS_PUBLISH_INTERVAL_SECONDS:
            self._last_windows_at = now
            self._publish_windows(client, result)

        if publish_epoch and result.decision:
            self._publish_state(client, result.decision, result=result)

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

        payload = {
            "ts": int(time.time() * 1000),
            "live": live,
            "engine": "compute",
            "emotion": decision.display_label,
            "emotion_baseline": decision.baseline_label,
            "labels_agree": decision.labels_agree,
            "arousal_10": decision.display_arousal_10,
            "arousal_pct": round(decision.display_arousal_10 * 10.0, 1),
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
            payload["window_used"] = "w30"
            payload["data_quality"] = result.best.quality
            payload["ibi_buffer_size"] = result.ibi_buffer_size
        if multi:
            for suffix, metrics in (
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
            self._apply_live_arousal_smoothing(payload, decision)
        else:
            # No decision yet means we have no fresh integer arousal to publish.
            # Reset hysteresis state so a later run does not flip from a stale
            # value carried across the gap.
            self._live_displayed_a10 = None
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            payload = {
                "ts": int(time.time() * 1000),
                "live": True,
                "engine": "compute",
                "window_used": "w30",
                "data_quality": result.best.quality,
                "ibi_buffer_size": result.ibi_buffer_size,
                "motion_class": result.motion_class,
                "baseline_ready": result.baseline_ready,
                "arousal_10": None,
                "stress_index": None,
                "rmssd": None,
                "mean_hr": None,
                "window_sec": PRIMARY_DECISION_WINDOW_SECONDS,
            }
        client.publish("biofizic/state/live", json.dumps(payload), qos=0)

    def _apply_live_arousal_smoothing(
        self, payload: dict, decision: PhysiologyDecision
    ) -> None:
        """
        Apply streak-based hysteresis to the integer arousal_10 reported on
        biofizic/state/live.

        The pipeline recomputes Kubios SI every second on a rolling 30 s buffer.
        A single new IBI can shift SI just enough to cross a zone boundary,
        which flips the rounded arousal_10 by one. Without hysteresis the watch
        UI alternates 2-3-2-3 every second even when the underlying signal is
        stable. We only adopt a new displayed integer after it has been seen
        for LIVE_AROUSAL_HYSTERESIS_TICKS consecutive ticks; otherwise we keep
        the previous one. The label is recomputed from the displayed integer so
        the text on the watch stays consistent with the number.
        """
        raw_a10 = int(decision.display_arousal_10)
        displayed = self._update_live_arousal_hysteresis(raw_a10)
        payload["arousal_10"] = displayed
        payload["arousal_pct"] = round(displayed * 10.0, 1)
        payload["emotion"] = arousal_scale_10_to_label(displayed)
        # Keep the unsmoothed value visible for Grafana and debugging so we
        # can see when the hysteresis is hiding a real transition.
        payload["arousal_10_raw"] = raw_a10

    def _update_live_arousal_hysteresis(self, raw_a10: int) -> int:
        if self._live_displayed_a10 is None:
            self._live_displayed_a10 = raw_a10
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            return raw_a10
        if raw_a10 == self._live_displayed_a10:
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            return self._live_displayed_a10
        if raw_a10 == self._live_candidate_a10:
            self._live_candidate_streak += 1
        else:
            self._live_candidate_a10 = raw_a10
            self._live_candidate_streak = 1
        if self._live_candidate_streak >= LIVE_AROUSAL_HYSTERESIS_TICKS:
            self._live_displayed_a10 = raw_a10
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
        return self._live_displayed_a10

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
        result: MultiWindowResult | None = None,
    ) -> None:
        """
        Publish the epoch decision (every 30 s) on biofizic/state. The message
        is retained so a watch reconnecting between epochs can bootstrap from
        the last known decision without waiting up to 30 s for the next one.
        """
        payload = self._decision_payload(decision, live=False, result=result)
        client.publish("biofizic/state", json.dumps(payload), qos=1, retain=True)

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
