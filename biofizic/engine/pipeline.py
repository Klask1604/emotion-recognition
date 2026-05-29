"""Unified physiology compute pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.decision import DecisionState, decide
from biofizic.engine.signal_quality import SignalQualityState, update_and_score
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
        self.decision_state = DecisionState()
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

        # Decision now runs on w60 instead of w30: empirically the w30 RMSSD
        # has ~1.8x the std-dev and unrealistic single-beat spikes (max 159
        # ms on quiet wear-time vs 104 ms on w90). w60 covers a full
        # respiratory cycle so RSA averages out, and one bad beat contributes
        # only 1/(78 beats) ~ 1.3% instead of 1/40 ~ 2.5%. We fall back to
        # w30 when w60 isn't yet computable (first 60 s of recording).
        primary = multi.window_60_seconds or multi.window_30_seconds
        primary_window_label = (
            "w60" if multi.window_60_seconds is not None else "w30"
        )

        # Motion energy over the same window (cardiac-band acc reported by the
        # watch, median over the buffer); artifact rate from the primary window.
        motion_energy = self.motion_buffer.median_in_last_ms(
            PRIMARY_DECISION_WINDOW_SECONDS * 1000, end_ms=end_ms
        ) or 0.0
        artifact_rate = primary.artifact_rate if primary is not None else 0.0
        # Pass has_signal=False when the primary window has no beats; without
        # this flag artifact_rate=0 would be treated as "perfect" and
        # signal_quality would report ~0.97 on an empty IBI buffer, masking
        # silent watch periods as high-confidence (the source of the long-
        # standing fake-baseline-ready bug).
        has_signal = (
            primary is not None and primary.beat_count >= MIN_BEATS_FOR_ANY_HRV
        )
        quality = update_and_score(
            motion_energy=motion_energy,
            artifact_rate=artifact_rate,
            state=self.quality_state,
            has_signal=has_signal,
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
        # Gate on the primary window (w60 when available, w30 fallback).
        primary_window_result = w60 if primary is multi.window_60_seconds else w30
        if (
            primary_window_result.quality != "unavailable"
            and primary is not None
            and primary.beat_count >= MIN_BEATS_FOR_ANY_HRV
        ):
            decision = decide(
                primary=primary,
                multi=multi,
                sensor=sensor,
                quality=quality,
                baseline=self.baseline,
                state=self.decision_state,
                publish_epoch=publish_epoch,
            )
            if publish_epoch:
                self.state.epoch_count += 1
                try:
                    log.info(format_decision_block(decision))
                except Exception as exc:  # noqa: BLE001
                    log.warning("decision log formatting failed: %s", exc)

        self.state.last_decision_at = now_ts
        # Expose the actual primary window in `best` so dashboards and the
        # watch payload don't lie about which window drove the verdict.
        best = primary_window_result
        return MultiWindowResult(
            ts=now_ts,
            w30=w30,
            w60=w60,
            w90=w90,
            best=best,
            best_window_label=primary_window_label,
            decision=decision,
            ibi_buffer_size=buf_size,
            motion_state=quality.motion_state,
            signal_quality=quality.quality,
            artifact_rate=quality.artifact_rate,
            baseline_ready=self.baseline.is_ready,
        )

    def reset_baseline(self, reported_arousal: float | None = None) -> None:
        self.baseline.reset_for_recalibration(reported_arousal)
        self.decision_state.reset()
