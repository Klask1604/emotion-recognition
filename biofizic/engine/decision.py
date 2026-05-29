"""
Decision core: HRV features + quality + baseline → PhysiologyDecision.

This module replaces what used to live in four separate files
(`state_estimator.py`, `cusum.py`, `decision_gate.py`, plus the
population/personal mapping previously in `arousal_mapper.py`). The decision
flow is linear and benefits from being read top-to-bottom in one place:

    fusion:    z_fused = Q · z_hrv + (1 - Q) · z_hr
    smoothing: scalar Kalman with measurement variance = BASE / Q
    alert:     one-sided CUSUM on the filtered z (Page 1954)
    mapping:   personal Φ(z + offset) → arousal_10  (if baseline ready)
               OR Kubios population zone of SI         (preliminary fallback)

State carried between epochs (Kalman x/P, CUSUM accumulator and latch) lives
in a single dataclass `DecisionState`. Math helpers (normal CDF, Kubios zone
lookup, personal mapping) are private functions in this file.

The public API is one function: `decide(...)` plus `DecisionState`.

`PhysiologyDecision` (the shape served to the watch and the dashboards)
remains identical — this refactor is purely internal consolidation.
"""

from __future__ import annotations

from dataclasses import dataclass

from biofizic.compute_features.results import (
    HrvMetrics,
    MultiWindowHrvResult,
    PhysiologyDecision,
)
from biofizic.config import (
    CHANNEL_HR_DOMINANT_BELOW,
    CHANNEL_HRV_DOMINANT_ABOVE,
    CUSUM_SLACK_K,
    CUSUM_THRESHOLD_H,
    HR_CHANNEL_CONFIDENCE,
    KALMAN_MEAS_VAR_BASE,
    KALMAN_PROCESS_VAR,
    KALMAN_QUALITY_FLOOR,
    PRELIMINARY_CONFIDENCE_CAP,
    TEMP_CHANNEL_MAX_WEIGHT,
)
from biofizic.engine.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    personal_arousal_10,
    population_arousal_10,
)
from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.channels.temperature import (
    SkinTemperatureChannelState,
    evaluate_skin_temperature,
)
from biofizic.engine.signal_quality import SignalQuality
from biofizic.ingestion.messages import SensorBatchMessage


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class DecisionState:
    """Per-session mutable state owned by the decision module: Kalman estimate
    + variance, plus CUSUM accumulator + latched alert. Single dataclass so
    `pipeline.PhysiologyPipeline` carries one state object instead of three."""

    # Scalar Kalman on the personal stress z. x=0 means "at personal baseline".
    estimator_x: float = 0.0
    estimator_P: float = 1.0
    estimator_process_var: float = KALMAN_PROCESS_VAR

    # One-sided CUSUM on the filtered z. Latches True until the accumulator
    # decays back to zero, so the alert has built-in hysteresis (Page 1954).
    cusum_slack_k: float = CUSUM_SLACK_K
    cusum_threshold_h: float = CUSUM_THRESHOLD_H
    cusum_s: float = 0.0
    cusum_alert: bool = False

    def reset(self) -> None:
        self.estimator_x = 0.0
        self.estimator_P = 1.0
        self.cusum_s = 0.0
        self.cusum_alert = False


# ── Private math: Kalman, CUSUM, mappings ────────────────────────────────────

def _kalman_update(state: DecisionState, z_measured: float, quality: float) -> tuple[float, float]:
    """Fold one epoch's z into the Kalman estimate. Returns (x_filtered, gain).

    Measurement variance scales as BASE / max(Q, FLOOR): a low-quality epoch
    yields a large variance, tiny Kalman gain, so the estimate barely moves.
    This unifies the signal-quality gate and the older "hold last value" patch
    into one principled filter.
    """
    q = max(float(quality), KALMAN_QUALITY_FLOOR)
    r = KALMAN_MEAS_VAR_BASE / q
    state.estimator_P += state.estimator_process_var
    gain = state.estimator_P / (state.estimator_P + r)
    state.estimator_x += gain * (float(z_measured) - state.estimator_x)
    state.estimator_P *= 1.0 - gain
    return state.estimator_x, gain


def _cusum_update(state: DecisionState, z: float) -> bool:
    """One-sided CUSUM:   S_t = max(0, S_{t-1} + (z_t - k))
    Alerts when S_t > h; the alert latches until S_t decays back to 0."""
    state.cusum_s = max(0.0, state.cusum_s + (float(z) - state.cusum_slack_k))
    if state.cusum_s > state.cusum_threshold_h:
        state.cusum_alert = True
    elif state.cusum_s == 0.0:
        state.cusum_alert = False
    return state.cusum_alert


# ── Multi-channel fusion ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FusionChannel:
    """One arousal channel feeding the weighted-mean fusion.

    name:       diagnostic label ("hrv", "hr", "temp", "resp").
    z:          the channel's personal z-score (>0 = more aroused).
    weight:     how much this channel contributes to the fused z. For HRV this
                is the signal quality Q; for HR it is (1 - Q). New channels
                (temp, resp) enter with weight 0 until their own quality gate is
                wired in, which makes them an exact no-op against the current
                pipeline (see _fuse).
    confidence: the channel's own confidence, blended into fused_confidence with
                the same weights.
    """

    name: str
    z: float
    weight: float
    confidence: float


def _fuse(channels: list[FusionChannel]) -> tuple[float, float]:
    """Weighted mean of the channel z-scores and confidences.

        z_fused    = Σ wᵢ·zᵢ    / Σ wᵢ
        confidence = Σ wᵢ·confᵢ / Σ wᵢ

    With only HRV (weight Q) and HR (weight 1-Q) the denominator is exactly 1,
    so this reduces bit-for-bit to the previous
        z_fused = Q·z_hrv + (1-Q)·z_hr
    The normalisation only changes anything once a third channel adds weight,
    at which point every channel is re-weighted to keep z_fused on the same
    z-score scale. Returns (0, 0) if no channel carries weight."""
    total_w = sum(c.weight for c in channels)
    if total_w <= 0.0:
        return 0.0, 0.0
    z = sum(c.weight * c.z for c in channels) / total_w
    conf = sum(c.weight * c.confidence for c in channels) / total_w
    return z, conf


# ── Public decision entry point ──────────────────────────────────────────────

def decide(
    *,
    primary: HrvMetrics,
    multi: MultiWindowHrvResult,
    sensor: SensorBatchMessage | None,
    quality: SignalQuality,
    baseline: RestBaselineStore,
    state: DecisionState,
    publish_epoch: bool,
    temperature: "SkinTemperatureChannelState | None" = None,
) -> PhysiologyDecision:
    """One epoch tick: fuse channels, smooth via Kalman, gate the verdict.

    The decision flow:
      1. Compute personal z's (HRV + HR) — zero before baseline locks.
      2. Fuse:  z_fused = Q · z_hrv + (1 - Q) · z_hr
                 confidence = Q · q_hrv + (1 - Q) · q_hr
      3. Cap confidence in preliminary mode (no personal baseline yet).
      4. Fold z_fused into Kalman ONCE per epoch (1/30 Hz).
      5. Run CUSUM on the filtered z (only meaningful post-baseline).
      6. Map to arousal_10 via personal CDF (calibrated) or Kubios zone
         (preliminary).
      7. Build PhysiologyDecision with full diagnostic fields.
    """
    sdk_hr = sensor.heart_rate_bpm if sensor else 0.0

    # 1) Personal z-scores — both 0 until the baseline locks.
    stress_z = (
        baseline.stress_index_z_score(primary.kubios_stress_index)
        if baseline.is_ready
        else 0.0
    )
    hr_z = baseline.hr_z_score(sdk_hr) if baseline.is_ready else 0.0

    # 2) Motion-tolerant fusion: HRV is precise when still, HR is robust in
    # motion. Weight by signal quality so the verdict leans on HR during VR
    # activity instead of freezing. Built as a list of channels and combined by
    # a weighted mean (_fuse) so temp/resp channels can be appended later without
    # touching this math; with only HRV+HR the result is identical to the prior
    # z_fused = Q·z_hrv + (1-Q)·z_hr.
    hrv_weight = quality.quality
    hr_present = sdk_hr > 0.0 and baseline.is_ready
    hr_conf = HR_CHANNEL_CONFIDENCE if hr_present else 0.0
    channels = [
        FusionChannel("hrv", z=stress_z, weight=hrv_weight, confidence=quality.quality),
        FusionChannel("hr", z=hr_z, weight=1.0 - hrv_weight, confidence=hr_conf),
    ]
    # Optional skin-temperature channel (secondary arousal proxy). It enters
    # with weight = confidence · cap, so it is an EXACT no-op until its own
    # baseline locks and the ambient gate is open (confidence > 0) — which keeps
    # the HRV+HR-only result identical to before. The cap stops a slow signal
    # from ever dominating the verdict.
    temp_z = 0.0
    temp_confidence = 0.0
    if temperature is not None and sensor is not None:
        temp_eval = evaluate_skin_temperature(
            temperature,
            skin_temp_c=sensor.skin_temperature_c,
            ambient_temp_c=sensor.ambient_temperature_c,
        )
        temp_z = temp_eval.z
        temp_confidence = temp_eval.confidence
        temp_weight = temp_confidence * TEMP_CHANNEL_MAX_WEIGHT
        if temp_weight > 0.0:
            channels.append(
                FusionChannel("temp", z=temp_z, weight=temp_weight, confidence=temp_confidence)
            )
    z_fused, fused_confidence = _fuse(channels)

    # 3) Preliminary cap: a pre-baseline verdict comes from the Kubios
    # population zone, so it must not look as confident as a calibrated one.
    decision_fidelity = "calibrated" if baseline.is_ready else "preliminary"
    if not baseline.is_ready:
        fused_confidence = min(fused_confidence, PRELIMINARY_CONFIDENCE_CAP)

    if hrv_weight >= CHANNEL_HRV_DOMINANT_ABOVE:
        dominant_channel = "hrv"
    elif hrv_weight <= CHANNEL_HR_DOMINANT_BELOW:
        dominant_channel = "hr" if hr_present else "none"
    else:
        dominant_channel = "blend"

    # 4) Kalman — once per epoch, only when we have a personal anchor.
    # run() is called every second on the same rolling 30 s window; updating
    # every second would track that 1 Hz re-noise and make arousal jump.
    if publish_epoch and baseline.is_ready and primary.kubios_stress_index > 0:
        z_filtered, kalman_gain = _kalman_update(state, z_fused, fused_confidence)
    else:
        z_filtered, kalman_gain = state.estimator_x, 0.0

    # 5) Arousal mapping: personal CDF if calibrated, Kubios zone otherwise.
    offset_z = baseline.arousal_offset_z
    kubios_zone = kubios_zone_for_stress_index(primary.kubios_stress_index)
    kubios_label = kubios_zone.label
    if baseline.is_ready:
        arousal_10 = personal_arousal_10(z_filtered, offset_z)
        gate_mode = "personal_z"
    else:
        arousal_10 = population_arousal_10(primary.kubios_stress_index)
        gate_mode = "population_zone"
    display_label = arousal_scale_10_to_label(arousal_10)

    # 6) CUSUM on the filtered z (only after baseline locks; filtered z is
    # already quality-attenuated so artifact bursts can't push it).
    alert = _cusum_update(state, z_filtered) if baseline.is_ready else False
    if alert:
        gate_mode = "alert_confirmed"

    if baseline.is_ready:
        baseline_label = arousal_scale_10_to_label(
            personal_arousal_10(z_filtered, offset_z)
        )
    else:
        baseline_label = "pending"
    labels_agree = baseline.is_ready and kubios_label == baseline_label

    reasons = [
        f"kubios={kubios_label}",
        f"motion={quality.motion_state}",
        f"q={quality.quality:.2f}",
        f"artifact={quality.artifact_rate:.2f}",
    ]
    if alert:
        reasons.append("alert")
    if gate_mode not in ("personal_z", "population_zone") and gate_mode not in reasons:
        reasons.append(gate_mode)
    decision_reason = "|".join(reasons)

    baseline_si = float(baseline.baseline_stress_index or 0.0)

    return PhysiologyDecision(
        display_label=display_label,
        display_arousal_10=arousal_10,
        kubios_label=kubios_label,
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
        alert=alert,
        decision_reason=decision_reason,
        baseline_ready=baseline.is_ready,
        stress_index_z_filtered=z_filtered,
        kalman_gain=kalman_gain,
        hr_z_score=hr_z,
        hrv_weight=hrv_weight,
        decision_confidence=fused_confidence,
        dominant_channel=dominant_channel,
        decision_fidelity=decision_fidelity,
        multi_window=multi,
    )
