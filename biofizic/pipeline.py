"""Unified physiology compute pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

from biofizic.baseline import RestBaselineStore
from biofizic.config import motion_model_path
from biofizic.context_engine import ActivityContextEngine
from biofizic.decision.alert_confirmation import AlertConfirmationGate
from biofizic.decision.arousal_mapper import (
    baseline_z_score_to_label,
    kubios_zone_for_stress_index,
    stress_index_to_arousal,
    zone_is_elevated_or_higher,
)
from biofizic.decision.physiology_fusion import fuse_physiology_and_motion
from biofizic.decision.valence_mapper import compute_valence, finalize_affect_quadrant
from biofizic.features.hrv_metrics import (
    compute_hrv_from_mqtt_payload,
    warn_if_watch_server_rmssd_mismatch,
)
from biofizic.logging import format_decision_block
from biofizic.motion.motion_features import MotionFeatureVector
from biofizic.motion.motion_ml import MotionHarModel
from biofizic.types import (
    HrvMetrics,
    IbiBatchMessage,
    MultiWindowHrvResult,
    MultiWindowResult,
    PhysiologyDecision,
    PpgBatchMessage,
    SensorBatchMessage,
    WindowResult,
)
from biofizic.windows import MultiWindowProcessor, RollingIbiBuffer, RollingPpgBuffer

log = logging.getLogger("physiology_pipeline")

Z_PULSE_AMP_STALE_SEC = 90.0
_WINDOW_METRICS = {
    "w30": lambda m: m.window_30_seconds,
    "w60": lambda m: m.window_60_seconds,
    "w90": lambda m: m.window_90_seconds,
}


@dataclass
class PipelineState:
    """Mutable pipeline runtime state."""

    last_sensor: SensorBatchMessage | None = None
    elevated_streak: int = 0
    epoch_count: int = 0
    last_decision_at: float = 0.0
    last_epoch_publish_at: float = 0.0
    ppg_pipeline_last: dict = field(default_factory=dict)
    z_pulse_amp: float = 0.0
    z_pulse_amp_at: float = 0.0


class PhysiologyPipeline:
    """
    Server-side compute: buffers -> multi-window HRV -> baseline -> HAR -> decision.
    Never resets baseline on motion change.
    """

    def __init__(self) -> None:
        self.ibi_buffer = RollingIbiBuffer()
        self.ppg_buffer = RollingPpgBuffer()
        self.multi_window = MultiWindowProcessor()
        self.baseline = RestBaselineStore()
        self.har_model = MotionHarModel()
        self.activity_engine = ActivityContextEngine(
            motion_model_path(), persist_motion_model=True
        )
        self._activity = self.activity_engine.update(0.0, allow_quiet_learning=False)
        self.alert_gate = AlertConfirmationGate()
        self.state = PipelineState()

    def ingest_ibi_batch(self, batch: IbiBatchMessage) -> None:
        self.ibi_buffer.ingest_batch(batch)

    def ingest_ppg_batch(self, batch: PpgBatchMessage) -> None:
        self.ppg_buffer.ingest_batch(batch)

    def ingest_sensor_batch(self, batch: SensorBatchMessage) -> None:
        self.state.last_sensor = batch
        self._activity = self.activity_engine.update(
            batch.acceleration_rms, allow_quiet_learning=True
        )

    def ingest_ppg_hrv(self, z_pulse_amp: float, *, now: float | None = None) -> None:
        """Latest sympathetic proxy from ppg-processor (biofizic/ppg_hrv)."""
        self.state.z_pulse_amp = float(z_pulse_amp)
        self.state.z_pulse_amp_at = now if now is not None else time.time()

    def ingest_legacy_epoch(self, data: dict) -> None:
        """Backward compatibility for biofizic/epoch until watch sends batches only."""
        self.ibi_buffer.ingest_epoch_payload(data)
        sensor = SensorBatchMessage(
            timestamp_ms=int(data.get("ts") or time.time() * 1000),
            heart_rate_bpm=float(data.get("hr") or data.get("mean_hr") or 0),
            acceleration_rms=float(data.get("acc_rms") or 0),
            acceleration_p90=float(data.get("acc_p90") or 0),
            acceleration_std=float(data.get("acc_std") or 0),
            gyroscope_rms=float(data.get("gyro_rms") or 0),
            gyroscope_p90=float(data.get("gyro_p90") or 0),
            gyroscope_std=float(data.get("gyro_std") or 0),
            skin_temperature_c=float(data.get("skin_temp") or 0),
            ambient_temperature_c=float(data.get("ambient_temp") or 0),
            display_on=bool(data.get("displayOn", True)),
        )
        self.ingest_sensor_batch(sensor)

    def _fresh_z_pulse_amp(self, now_ts: float) -> float:
        if self.state.z_pulse_amp_at <= 0:
            return 0.0
        if now_ts - self.state.z_pulse_amp_at > Z_PULSE_AMP_STALE_SEC:
            return 0.0
        return self.state.z_pulse_amp

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

    @staticmethod
    def _cascade_best(
        results: dict[str, WindowResult],
    ) -> tuple[WindowResult, str]:
        for label in ("w90", "w60", "w30"):
            if results[label].quality == "full":
                return results[label], label
        for label in ("w60", "w30"):
            if results[label].quality == "partial":
                return results[label], label
        return results["w30"], "w30"

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
        publish_epoch: bool = False,
    ) -> MultiWindowResult:
        """Always returns a result; decision may be None when best window is unavailable."""
        now_ts = now if now is not None else time.time()
        sensor = self.state.last_sensor
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
        window_map = {"w30": w30, "w60": w60, "w90": w90}
        best, best_label = self._cascade_best(window_map)

        motion = self._predict_motion(sensor)

        if (
            motion.motion_class == "STILL"
            and multi.window_30_seconds is not None
            and multi.window_30_seconds.beat_count >= 2
        ):
            w30m = multi.window_30_seconds
            self.baseline.observe_still(w30m.rmssd_ms, w30m.kubios_stress_index)

        decision = None
        if best.quality != "unavailable":
            primary = _WINDOW_METRICS[best_label](multi)
            if primary is not None and primary.beat_count >= 2:
                decision = self._build_decision(
                    primary=primary,
                    multi=multi,
                    motion=motion,
                    publish_epoch=publish_epoch,
                    now_ts=now_ts,
                )

        self.state.last_decision_at = now_ts
        return MultiWindowResult(
            ts=now_ts,
            w30=w30,
            w60=w60,
            w90=w90,
            best=best,
            best_window_label=best_label,
            decision=decision,
            ibi_buffer_size=buf_size,
            motion_class=motion.motion_class,
            baseline_ready=self.baseline.is_ready,
        )

    def run_live_tick(self, *, now: float | None = None) -> PhysiologyDecision | None:
        """Backward-compatible wrapper; prefer run()."""
        return self.run(now=now).decision

    def run_epoch_tick(self, *, now: float | None = None) -> PhysiologyDecision | None:
        """Backward-compatible wrapper; prefer run(publish_epoch=True)."""
        return self.run(now=now, publish_epoch=True).decision

    def _build_decision(
        self,
        *,
        primary: HrvMetrics,
        multi: MultiWindowHrvResult,
        motion,
        publish_epoch: bool,
        now_ts: float,
    ) -> PhysiologyDecision:
        activity = self._activity

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

        zone = kubios_zone_for_stress_index(primary.kubios_stress_index)
        if zone_is_elevated_or_higher(zone):
            self.state.elevated_streak += 1
        else:
            self.state.elevated_streak = 0

        display_a10, display_label, kubios_label, reason = fuse_physiology_and_motion(
            kubios_stress_index=primary.kubios_stress_index,
            motion=motion,
            stress_index_z_score=stress_z,
            rmssd_z_score=rmssd_z,
            elevated_streak=self.state.elevated_streak,
        )

        arousal, _, _ = stress_index_to_arousal(primary.kubios_stress_index)
        arousal, display_a10, gate_mode = self.alert_gate.apply(
            kubios_stress_index=primary.kubios_stress_index,
            rmssd_ms=primary.rmssd_ms,
            stress_index_z_score=stress_z,
            rmssd_z_score=rmssd_z,
            activity=activity,
            arousal=arousal,
            arousal_scale_10=display_a10,
        )
        if gate_mode != "kubios_zone":
            reason = f"{reason}|{gate_mode}"

        z_pulse = self._fresh_z_pulse_amp(now_ts)
        valence = finalize_affect_quadrant(
            compute_valence(
                rmssd_z_score=rmssd_z,
                z_pulse_amp=z_pulse,
                motion_class=motion.motion_class,
                baseline_ready=self.baseline.is_ready,
            ),
            display_a10,
        )

        baseline_label = baseline_z_score_to_label(
            stress_z, baseline_ready=self.baseline.is_ready
        )
        baseline_si = float(self.baseline.baseline_stress_index or 0.0)
        labels_agree = (
            self.baseline.is_ready and kubios_label == baseline_label
        )

        if publish_epoch:
            self.state.epoch_count += 1

        decision = PhysiologyDecision(
            display_label=display_label,
            display_arousal_10=display_a10,
            kubios_label=kubios_label,
            baseline_label=baseline_label,
            labels_agree=labels_agree,
            kubios_stress_index=primary.kubios_stress_index,
            baseline_stress_index=baseline_si,
            stress_index_z_score=stress_z,
            rmssd_ms=primary.rmssd_ms,
            mean_heart_rate_bpm=primary.mean_heart_rate_bpm,
            motion_class=motion.motion_class,
            motion_confidence=motion.confidence,
            activity_mode=activity.mode.value,
            decision_reason=reason,
            baseline_ready=self.baseline.is_ready,
            multi_window=multi,
            valence_10=valence.valence_10,
            valence_label=valence.valence_label,
            affect_quadrant=valence.affect_quadrant,
            z_pulse_amp=z_pulse,
        )

        if publish_epoch:
            log.info(format_decision_block(decision))

        return decision

    def reset_baseline(self) -> None:
        self.baseline.reset_for_recalibration()

    def legacy_epoch_metrics(self, data: dict):
        metrics = compute_hrv_from_mqtt_payload(data)
        if metrics:
            warn_if_watch_server_rmssd_mismatch(data, metrics)
        return metrics
