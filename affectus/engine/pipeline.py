"""Unified physiology compute pipeline."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from affectus.shared.baseline import RestBaselineStore
from affectus.shared.valence_baseline import ValenceBaselineStore, ValenceSmoother
from affectus.shared.reactivity import ReactivityProfile
from affectus.devices.wrist.modules.temperature import SkinTemperatureChannelState
from affectus.engine.decision import DecisionState, decide
from affectus.shared.signal_quality import SignalQualityState, update_and_score
from affectus.logging import format_decision_block
from affectus.shared.hrv.results import (
    HrvMetrics,
    MultiWindowHrvResult,
    MultiWindowResult,
    PhysiologyDecision,
    WindowResult,
)
from affectus.ingestion.messages import (
    AcquisitionBatchMessage,
    IbiBatchMessage,
    SensorBatchMessage,
)
from affectus.config import (
    HRV_LOOKBACK_MS,
    MIN_BEATS_FOR_ANY_HRV,
    PRIMARY_DECISION_WINDOW_SECONDS,
)
from affectus.shared.hrv.windows import MultiWindowProcessor, RollingIbiBuffer, RollingSensorBuffer

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
    physical cause that HAR only approximated (body motion corrupting the PPG)
    is measured directly via a signal-quality gate: cardiac-band acceleration
    energy plus the IBI artifact rate (see decision/signal_quality.py).
    """

    def __init__(self) -> None:
        self.ibi_buffer = RollingIbiBuffer()
        self.motion_buffer = RollingSensorBuffer()
        self.multi_window = MultiWindowProcessor()
        self.baseline = RestBaselineStore()
        # Skin-temperature arousal channel: persisted in its own file so it
        # never touches rest_baseline.json. Resting samples feed it on the same
        # quality gate as the HRV baseline.
        self.temperature = SkinTemperatureChannelState(persist=True)
        # Personal valence baseline: recenters the cross-device WESAD model on
        # this subject (neutral = measured resting median, not assumed 0). Fed by
        # the compute service on the same resting gate as the HRV baseline.
        self.valence_baseline = ValenceBaselineStore(persist=True)
        # 60 s rolling mean of personal valence: reports the background emotional
        # state (mood) rather than each noisy 1 s epoch. Matched to the arousal
        # decision window (w60) so the circumplex has uniform inertia on both axes.
        self.valence_smoother = ValenceSmoother(window_s=60.0)
        # EEVR- and CASE-trained valence models get their OWN personal baseline +
        # smoother (own file, own neutral), so each is re-centered on the subject
        # exactly like WESAD and the comparison dashboard shows calibrated emotion
        # for all three, not raw valence. Same resting gate feeds all three.
        from affectus.config import data_dir
        self.valence_baseline_eevr = ValenceBaselineStore(
            path=data_dir() / "valence_baseline_eevr.json")
        self.valence_smoother_eevr = ValenceSmoother(window_s=60.0)
        self.valence_baseline_case = ValenceBaselineStore(
            path=data_dir() / "valence_baseline_case.json")
        self.valence_smoother_case = ValenceSmoother(window_s=60.0)
        # One-time subject reactivity profile: scales the valence dead-band.
        self.reactivity = ReactivityProfile(persist=True)
        # While an ECG calibration is in progress, PPG must NOT lock the baseline
        # (it would win the race before the user puts their finger on the button).
        # 0 = no ECG window; otherwise the epoch-clock deadline past which PPG
        # takes over as fallback. Set by the compute service.
        self.ecg_calibration_until: float = 0.0
        # ECG calibration beats accumulated this hold. HRV is computed directly
        # from these (no 60 s rolling window), so the baseline can lock within the
        # 45 s finger-hold instead of waiting for a full window to fill.
        self._ecg_cal_beats: list = []
        self.quality_state = SignalQualityState()
        self.decision_state = DecisionState()
        self.state = PipelineState()

    def ingest_ibi_batch(self, batch: IbiBatchMessage) -> None:
        self.ibi_buffer.ingest_batch(batch)

    def reset_ecg_calibration(self) -> None:
        """Clear the accumulated ECG beats at the start of a new finger-hold."""
        self._ecg_cal_beats = []

    def ingest_ecg_calibration(self, batch: IbiBatchMessage, *, now: float) -> bool:
        """Feed ECG-derived IBI into the arousal baseline during calibration.

        HRV is computed DIRECTLY from the ECG beats accumulated this hold (no 60 s
        rolling window), so the baseline locks within the 45 s finger-hold rather
        than waiting for a full HRV window to fill. ECG (a deliberate finger-hold)
        is motion-immune and the subject is still by definition, so this BYPASSES
        the motion/quality gate the PPG path needs. The 1 s spacing gate inside
        observe_resting is kept (pass `now`) so the locked baseline's MAD stays
        meaningful: a sample is accepted ~once per real second of contact, 8 of
        them lock it in ~8 s. Returns whether the baseline has locked."""
        from affectus.shared.hrv.metrics import compute_hrv_from_entries
        from affectus.ingestion.messages import InterbeatIntervalEntry

        for ms, ts in zip(batch.intervals_ms, batch.timestamps_ms):
            self._ecg_cal_beats.append(InterbeatIntervalEntry(interval_ms=ms, timestamp_ms=ts))
        # Keep a bounded recent window of beats so the HRV stats reflect the hold,
        # not stale data (a comfortable ~90 s of beats at any heart rate).
        if len(self._ecg_cal_beats) > 200:
            self._ecg_cal_beats = self._ecg_cal_beats[-200:]
        if len(self._ecg_cal_beats) < MIN_BEATS_FOR_ANY_HRV:
            return self.baseline.is_ready
        hrv = compute_hrv_from_entries(self._ecg_cal_beats)
        if hrv is None or hrv.kubios_stress_index <= 0 or hrv.rmssd_ms <= 0:
            return self.baseline.is_ready
        self.baseline.observe_resting(
            hrv.rmssd_ms,
            hrv.kubios_stress_index,
            heart_rate_bpm=hrv.mean_heart_rate_bpm,
            now=now,
            min_spacing_s=1.0,
        )
        return self.baseline.is_ready

    def ingest_acquisition(self, batch: AcquisitionBatchMessage) -> None:
        """Atomic ingest of a v2 watch batch. Wraps it into a device-agnostic
        SensorFrame and delegates, so every wearable enters through one path."""
        from affectus.devices.wrist.adapter import frame_from_acquisition_v2

        self.ingest_frame(frame_from_acquisition_v2(batch))

    def ingest_frame(self, frame: "SensorFrame") -> None:
        """Capability-agnostic ingest: route the canonical slots of any frame
        into the rolling buffers. Only signals the frame actually carries are
        ingested (e.g. no MOTION capability -> motion buffer untouched)."""
        from affectus.contract.capabilities import Capability

        if frame.has(Capability.IBI) and frame.ibi is not None:
            self.ingest_ibi_batch(frame.ibi)
        # The legacy decision path still reads SensorBatchMessage; keep it fed
        # from the frame's raw message when present (wrist v2), else synthesise.
        if frame.raw is not None:
            self.ingest_sensor_batch(frame.raw.to_sensor_batch())
        if frame.has(Capability.MOTION) and frame.motion_energy is not None:
            self.motion_buffer.ingest(frame.timestamp_ms, frame.motion_energy)

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
        # During an active ECG-calibration window, PPG is suppressed so the clean
        # ECG IBI builds the baseline (PPG would otherwise lock it first). Past the
        # window deadline (user never held the finger), PPG resumes as fallback.
        ecg_window_active = now_ts < self.ecg_calibration_until
        if (
            not ecg_window_active
            and quality.usable
            and quality.motion_state == "still"
            and primary is not None
            and primary.beat_count >= MIN_BEATS_FOR_ANY_HRV
        ):
            self.baseline.observe_resting(
                primary.rmssd_ms,
                primary.kubios_stress_index,
                heart_rate_bpm=sdk_hr,
                now=now_ts,
            )
            # Feed the temperature baseline on the SAME resting gate, so its
            # personal reference is built from the same quiet epochs.
            if sensor is not None and sensor.skin_temperature_c > 0:
                self.temperature.observe_resting(
                    sensor.skin_temperature_c,
                    sensor.ambient_temperature_c,
                    now=now_ts,
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
                temperature=self.temperature,
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

    def reset_baseline(
        self,
        reported_arousal: float | None = None,
        reported_valence: float | None = None,
        reactivity: str | None = None,
    ) -> None:
        self.baseline.reset_for_recalibration(reported_arousal)
        self.temperature.reset()
        self.decision_state.reset()
        # Recalibration must also re-anchor valence: the personal neutral and the
        # smoother are subject/session-specific, so a recalibrate re-measures them
        # from the new resting epochs (otherwise valence stays on the old neutral).
        # reported_valence feeds the polarity sign-guard (does not move neutral).
        self.valence_baseline.reset_for_recalibration(reported_valence)
        self.valence_smoother.reset()
        # The EEVR/CASE comparison baselines re-anchor on the same recalibration.
        self.valence_baseline_eevr.reset_for_recalibration(reported_valence)
        self.valence_smoother_eevr.reset()
        self.valence_baseline_case.reset_for_recalibration(reported_valence)
        self.valence_smoother_case.reset()
        # Reactivity is a one-time subject profile; set only when the watch sends it.
        if reactivity is not None:
            self.reactivity.set(reactivity)
