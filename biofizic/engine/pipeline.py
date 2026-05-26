"""Unified physiology compute pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.arousal_mapper import (
    arousal_scale_10_to_label,
    personal_arousal_10,
)
from biofizic.engine.decision_gate import DecisionGateState, apply_decision_gate
from biofizic.engine.signal_quality import SignalQualityState, update_and_score
from biofizic.engine.state_estimator import StressStateEstimator
from biofizic.logging import format_decision_block
from biofizic.compute_features.results import (
    HrvMetrics,
    MultiWindowHrvResult,
    MultiWindowResult,
    PhysiologyDecision,
    WindowResult,
)
from biofizic.ingestion.messages import (
    AcquisitionBatchMessage,
    IbiBatchMessage,
    SensorBatchMessage,
)
from biofizic.config import (
    CHANNEL_HR_DOMINANT_BELOW,
    CHANNEL_HRV_DOMINANT_ABOVE,
    HR_CHANNEL_CONFIDENCE,
    HRV_LOOKBACK_MS,
    MIN_BEATS_FOR_ANY_HRV,
    PRIMARY_DECISION_WINDOW_SECONDS,
)
from biofizic.compute_features.windows import MultiWindowProcessor, RollingIbiBuffer, RollingSensorBuffer

log = logging.getLogger("physiology_pipeline")


@dataclass
class PipelineState:
    """Mutable pipeline runtime state."""

    last_sensor: SensorBatchMessage | None = None
    epoch_count: int = 0
    last_decision_at: float = 0.0


class PhysiologyPipeline:
    """
    Server-side compute: buffers -> multi-window HRV -> baseline -> signal
    quality -> decision. Never resets baseline on motion change.

    The earlier WISDM HAR classifier, motion calibrator and context engine have
    been removed. Motion is no longer classified into activities; instead the
    physical cause that HAR only approximated (wrist motion corrupting the PPG)
    is measured directly via a signal-quality gate: cardiac-band acceleration
    energy plus the IBI artifact rate (see decision/signal_quality.py).
    """

    def __init__(self) -> None:
        self.ibi_buffer = RollingIbiBuffer()
        self.motion_buffer = RollingSensorBuffer()
        self.multi_window = MultiWindowProcessor()
        self.baseline = RestBaselineStore()
        self.quality_state = SignalQualityState()
        self.estimator = StressStateEstimator()
        self.decision_gate_state = DecisionGateState()
        self.state = PipelineState()

    def ingest_ibi_batch(self, batch: IbiBatchMessage) -> None:
        self.ibi_buffer.ingest_batch(batch)

    def ingest_acquisition(self, batch: AcquisitionBatchMessage) -> None:
        """Atomic ingest: IBI + sensor stats share ts_anchor."""
        self.ingest_ibi_batch(batch.to_ibi_batch())
        self.ingest_sensor_batch(batch.to_sensor_batch())
        self.motion_buffer.ingest(batch.timestamp_anchor_ms, batch.motion_energy())

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

        all_entries = self.ibi_buffer.entries_in_last_ms(HRV_LOOKBACK_MS, end_ms=end_ms)
        buf_size = len(all_entries)

        if buf_size >= MIN_BEATS_FOR_ANY_HRV:
            multi = self.multi_window.compute(all_entries, end_timestamp_ms=end_ms)
        else:
            multi = MultiWindowHrvResult(None, None, None, None)

        w30 = self._window_result_from_metrics(multi.window_30_seconds)
        w60 = self._window_result_from_metrics(multi.window_60_seconds)
        w90 = self._window_result_from_metrics(multi.window_90_seconds)

        primary = multi.window_30_seconds

        # Motion energy over the same window (cardiac-band acc reported by the
        # watch, median over the buffer); artifact rate from the primary window.
        motion_energy = self.motion_buffer.median_in_last_ms(
            PRIMARY_DECISION_WINDOW_SECONDS * 1000, end_ms=end_ms
        ) or 0.0
        artifact_rate = primary.artifact_rate if primary is not None else 0.0
        quality = update_and_score(
            motion_energy=motion_energy,
            artifact_rate=artifact_rate,
            state=self.quality_state,
        )

        sdk_hr = sensor.heart_rate_bpm if sensor else 0.0

        # Lock / update the personal baseline only on high-quality resting epochs.
        if (
            quality.usable
            and quality.motion_state == "still"
            and primary is not None
            and primary.beat_count >= MIN_BEATS_FOR_ANY_HRV
        ):
            self.baseline.observe_resting(
                primary.rmssd_ms, primary.kubios_stress_index, heart_rate_bpm=sdk_hr
            )

        decision = None
        if (
            w30.quality != "unavailable"
            and primary is not None
            and primary.beat_count >= MIN_BEATS_FOR_ANY_HRV
        ):
            decision = self._build_decision(
                primary=primary,
                multi=multi,
                quality=quality,
                sdk_hr=sdk_hr,
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
            motion_state=quality.motion_state,
            signal_quality=quality.quality,
            artifact_rate=quality.artifact_rate,
            baseline_ready=self.baseline.is_ready,
        )

    def _build_decision(
        self,
        *,
        primary: HrvMetrics,
        multi: MultiWindowHrvResult,
        quality,
        sdk_hr: float,
        publish_epoch: bool,
        now_ts: float,
    ) -> PhysiologyDecision:
        stress_z = (
            self.baseline.stress_index_z_score(primary.kubios_stress_index)
            if self.baseline.is_ready
            else 0.0
        )
        hr_z = self.baseline.hr_z_score(sdk_hr) if self.baseline.is_ready else 0.0

        # Motion-tolerant fusion: blend the HRV-based z (precise when still) with
        # the HR-based z (robust in motion), weighted by signal quality. In a VR
        # active scene (low Q) the verdict leans on HR -> genuine high arousal,
        # instead of freezing. The fused measurement keeps a confidence floor
        # from the HR channel so the Kalman still updates during motion.
        hrv_weight = quality.quality
        z_fused = hrv_weight * stress_z + (1.0 - hrv_weight) * hr_z
        # Honest, multi-channel confidence: when HRV quality (hrv_weight) is low
        # in motion, the HR channel still carries the verdict, so confidence
        # floors near HR_CHANNEL_CONFIDENCE instead of collapsing to 0. If the
        # SDK gives no HR (sdk_hr<=0) there is no robust channel -> no floor.
        hr_present = sdk_hr > 0.0 and self.baseline.is_ready
        hr_conf = HR_CHANNEL_CONFIDENCE if hr_present else 0.0
        fused_confidence = hrv_weight * quality.quality + (1.0 - hrv_weight) * hr_conf
        if hrv_weight >= CHANNEL_HRV_DOMINANT_ABOVE:
            dominant_channel = "hrv"
        elif hrv_weight <= CHANNEL_HR_DOMINANT_BELOW:
            dominant_channel = "hr" if hr_present else "none"
        else:
            dominant_channel = "blend"

        # Fold z_fused into the Kalman smoother ONCE PER EPOCH (publish_epoch).
        # run() is called every second on the same rolling 30 s window; updating
        # every second would track that 1 Hz re-noise and make arousal jump.
        if publish_epoch and self.baseline.is_ready and primary.kubios_stress_index > 0:
            z_filtered, kalman_gain = self.estimator.update(z_fused, fused_confidence)
        else:
            z_filtered, kalman_gain = self.estimator.value(), 0.0

        offset_z = self.baseline.arousal_offset_z
        gate = apply_decision_gate(
            kubios_stress_index=primary.kubios_stress_index,
            stress_index_z_filtered=z_filtered,
            quality=quality,
            baseline_ready=self.baseline.is_ready,
            gate_state=self.decision_gate_state,
            arousal_offset_z=offset_z,
        )
        reason = gate.decision_reason
        if gate.gate_mode not in ("personal_z", "population_zone") and gate.gate_mode not in reason:
            reason = f"{reason}|{gate.gate_mode}"

        baseline_si = float(self.baseline.baseline_stress_index or 0.0)
        if self.baseline.is_ready:
            baseline_label = arousal_scale_10_to_label(
                personal_arousal_10(z_filtered, offset_z)
            )
        else:
            baseline_label = "pending"
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
            motion_state=quality.motion_state,
            signal_quality=quality.quality,
            artifact_rate=quality.artifact_rate,
            motion_energy=quality.motion_energy,
            alert=gate.alert,
            decision_reason=reason,
            baseline_ready=self.baseline.is_ready,
            stress_index_z_filtered=z_filtered,
            kalman_gain=kalman_gain,
            hr_z_score=hr_z,
            hrv_weight=hrv_weight,
            decision_confidence=fused_confidence,
            dominant_channel=dominant_channel,
            multi_window=multi,
        )

        if publish_epoch:
            # Logging must never take down the decision pipeline.
            try:
                log.info(format_decision_block(decision))
            except Exception as exc:  # noqa: BLE001
                log.warning("decision log formatting failed: %s", exc)

        return decision

    def reset_baseline(self, reported_arousal: float | None = None) -> None:
        self.baseline.reset_for_recalibration(reported_arousal)
        self.estimator.reset()
        self.decision_gate_state.cusum.reset()
