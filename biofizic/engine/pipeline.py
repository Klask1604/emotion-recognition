"""Unified physiology compute pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from biofizic.baseline import RestBaselineStore
from biofizic.decision.arousal_mapper import baseline_z_score_to_label
from biofizic.decision.decision_gate import DecisionGateState, apply_decision_gate
from biofizic.logging import format_decision_block
from biofizic.motion.motion_features import MotionFeatureVector
from biofizic.motion.motion_ml import MotionHarModel
from biofizic.types import (
    HrvMetrics,
    AcquisitionBatchMessage,
    IbiBatchMessage,
    MultiWindowHrvResult,
    MultiWindowResult,
    PhysiologyDecision,
    SensorBatchMessage,
    WindowResult,
)
from biofizic.windows import MultiWindowProcessor, RollingIbiBuffer

log = logging.getLogger("physiology_pipeline")


@dataclass
class PipelineState:
    """Mutable pipeline runtime state."""

    last_sensor: SensorBatchMessage | None = None
    epoch_count: int = 0
    last_decision_at: float = 0.0


class PhysiologyPipeline:
    """
    Server-side compute: buffers -> multi-window HRV -> baseline -> HAR -> decision.
    Never resets baseline on motion change.

    Motion is owned by a single source: MotionHarModel (WISDM HAR with a
    rule-based fallback when no model file is shipped). The earlier
    context_engine and adaptive_motion baseline classes were redundant and
    have been removed; `activity_mode` is now an alias for `motion_class`
    to keep downstream Grafana queries working.
    """

    def __init__(self) -> None:
        self.ibi_buffer = RollingIbiBuffer()
        self.multi_window = MultiWindowProcessor()
        self.baseline = RestBaselineStore()
        self.har_model = MotionHarModel()
        self.decision_gate_state = DecisionGateState()
        self.state = PipelineState()

    def ingest_ibi_batch(self, batch: IbiBatchMessage) -> None:
        self.ibi_buffer.ingest_batch(batch)

    def ingest_acquisition(self, batch: AcquisitionBatchMessage) -> None:
        """Atomic ingest: IBI + sensor stats share ts_anchor."""
        self.ingest_ibi_batch(batch.to_ibi_batch())
        self.ingest_sensor_batch(batch.to_sensor_batch())

    def ingest_sensor_batch(self, batch: SensorBatchMessage) -> None:
        self.state.last_sensor = batch

    @staticmethod
    def _window_result_from_metrics(metrics: HrvMetrics | None) -> WindowResult:
        if metrics is None or metrics.beat_count < 2:
            return WindowResult.unavailable()
        quality = "full" if metrics.is_valid else "partial"
        return WindowResult(
            rmssd_ms=metrics.rmssd_ms,
            sdnn_ms=metrics.sdnn_ms,
            pnn50_pct=metrics.pnn50_percent,
            kubios_stress_index=metrics.kubios_stress_index,
            mean_hr_bpm=metrics.mean_heart_rate_bpm,
            quality=quality,
            ibi_count=metrics.beat_count,
            covered_seconds=metrics.covered_seconds,
        )

    def _predict_motion(self, sensor: SensorBatchMessage | None):
        motion_feat = MotionFeatureVector.from_epoch_dict(
            {
                "acc_rms": sensor.acceleration_rms if sensor else 0,
                "acc_p90": sensor.acceleration_p90 if sensor else 0,
                "acc_std": sensor.acceleration_std if sensor else 0,
                "gyro_rms": sensor.gyroscope_rms if sensor else 0,
                "gyro_p90": sensor.gyroscope_p90 if sensor else 0,
                "gyro_std": sensor.gyroscope_std if sensor else 0,
            }
        )
        return self.har_model.predict(motion_feat)

    def run(
        self,
        *,
        now: float | None = None,
        end_timestamp_ms: int | None = None,
        publish_epoch: bool = False,
    ) -> MultiWindowResult:
        """Always returns a result; decision may be None when w30 is unavailable."""
        now_ts = now if now is not None else time.time()
        sensor = self.state.last_sensor
        end_ms = end_timestamp_ms
        if end_ms is None:
            end_ms = int(sensor.timestamp_ms) if sensor else int(now_ts * 1000)

        all_entries = self.ibi_buffer.entries_in_last_ms(90_000, end_ms=end_ms)
        buf_size = len(all_entries)

        if buf_size >= 2:
            multi = self.multi_window.compute(all_entries, end_timestamp_ms=end_ms)
        else:
            multi = MultiWindowHrvResult(None, None, None, None)

        w30 = self._window_result_from_metrics(multi.window_30_seconds)
        w60 = self._window_result_from_metrics(multi.window_60_seconds)
        w90 = self._window_result_from_metrics(multi.window_90_seconds)

        motion = self._predict_motion(sensor)

        if (
            motion.motion_class == "STILL"
            and multi.window_30_seconds is not None
            and multi.window_30_seconds.beat_count >= 2
        ):
            w30m = multi.window_30_seconds
            self.baseline.observe_still(w30m.rmssd_ms, w30m.kubios_stress_index)

        decision = None
        primary = multi.window_30_seconds
        if w30.quality != "unavailable" and primary is not None and primary.beat_count >= 2:
            decision = self._build_decision(
                primary=primary,
                multi=multi,
                motion=motion,
                sensor=sensor,
                publish_epoch=publish_epoch,
                now_ts=now_ts,
            )

        self.state.last_decision_at = now_ts
        return MultiWindowResult(
            ts=now_ts,
            w30=w30,
            w60=w60,
            w90=w90,
            best=w30,
            best_window_label="w30",
            decision=decision,
            ibi_buffer_size=buf_size,
            motion_class=motion.motion_class,
            baseline_ready=self.baseline.is_ready,
        )

    def _build_decision(
        self,
        *,
        primary: HrvMetrics,
        multi: MultiWindowHrvResult,
        motion,
        sensor: SensorBatchMessage | None,
        publish_epoch: bool,
        now_ts: float,
    ) -> PhysiologyDecision:
        acc_p90 = sensor.acceleration_p90 if sensor else 0.0

        stress_z = (
            self.baseline.stress_index_z_score(primary.kubios_stress_index)
            if self.baseline.is_ready
            else 0.0
        )
        rmssd_z = (
            self.baseline.rmssd_z_score(primary.rmssd_ms)
            if self.baseline.is_ready
            else 0.0
        )

        gate = apply_decision_gate(
            kubios_stress_index=primary.kubios_stress_index,
            rmssd_ms=primary.rmssd_ms,
            stress_index_z=stress_z,
            rmssd_z=rmssd_z,
            motion=motion,
            acc_p90=acc_p90,
            gate_state=self.decision_gate_state,
        )
        reason = gate.decision_reason
        if gate.gate_mode != "kubios_zone":
            reason = f"{reason}|{gate.gate_mode}"

        baseline_si = float(self.baseline.baseline_stress_index or 0.0)
        baseline_label = baseline_z_score_to_label(
            stress_z,
            baseline_ready=self.baseline.is_ready,
            baseline_si=baseline_si,
        )
        labels_agree = self.baseline.is_ready and gate.kubios_label == baseline_label

        if publish_epoch:
            self.state.epoch_count += 1

        decision = PhysiologyDecision(
            display_label=gate.display_label,
            display_arousal_10=gate.display_arousal_10,
            kubios_label=gate.kubios_label,
            baseline_label=baseline_label,
            labels_agree=labels_agree,
            kubios_stress_index=primary.kubios_stress_index,
            baseline_stress_index=baseline_si,
            stress_index_z_score=stress_z,
            rmssd_ms=primary.rmssd_ms,
            mean_heart_rate_bpm=primary.mean_heart_rate_bpm,
            motion_class=motion.motion_class,
            motion_confidence=motion.confidence,
            activity_mode=motion.motion_class,
            decision_reason=reason,
            baseline_ready=self.baseline.is_ready,
            multi_window=multi,
        )

        if publish_epoch:
            log.info(format_decision_block(decision))

        return decision

    def reset_baseline(self) -> None:
        self.baseline.reset_for_recalibration()
