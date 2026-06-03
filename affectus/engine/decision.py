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

from affectus.dsp.hrv.results import (
    HrvMetrics,
    MultiWindowHrvResult,
    PhysiologyDecision,
)
from affectus.config import (
    CHANNEL_HR_DOMINANT_BELOW,
    CHANNEL_HRV_DOMINANT_ABOVE,
    PRELIMINARY_CONFIDENCE_CAP,
)
from affectus.dsp.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    personal_arousal_10,
    population_arousal_10,
)
from affectus.dsp.baseline import RestBaselineStore
from affectus.dsp.temperature import SkinTemperatureChannelState
from affectus.dsp.signal_quality import SignalQuality
from affectus.io.messages import AcquisitionBatchMessage
from affectus.engine.channels import ChannelContext, build_channels

# The fusion math (state, Kalman, CUSUM, channel weighted-mean) is device-
# agnostic and lives in shared/. Re-exported here so existing importers of
# `engine.decision` (registry, pipeline, tests) keep working unchanged.
from affectus.dsp.fusion import (  # noqa: F401
    DecisionState,
    FusionChannel,
    _cusum_update,
    _fuse,
    _kalman_update,
)


# ── Public decision entry point ──────────────────────────────────────────────

def decide(
    *,
    primary: HrvMetrics,
    multi: MultiWindowHrvResult,
    sensor: AcquisitionBatchMessage | None,
    quality: SignalQuality,
    baseline: RestBaselineStore,
    state: DecisionState,
    publish_epoch: bool,
    temperature: "SkinTemperatureChannelState | None" = None,
    present: "frozenset | None" = None,
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
    # Whether the HR channel is meaningful (used below for dominant_channel).
    hr_present = sdk_hr > 0.0 and baseline.is_ready

    # 2) Motion-tolerant fusion: HRV is precise when still, HR is robust in
    # motion. Weight by signal quality so the verdict leans on HR during VR
    # activity instead of freezing. Built as a list of channels and combined by
    # a weighted mean (_fuse) so temp/resp channels can be appended later without
    # touching this math; with only HRV+HR the result is identical to the prior
    # z_fused = Q·z_hrv + (1-Q)·z_hr.
    hrv_weight = quality.quality
    # `present` is the set of sensors the device DECLARED at the handshake — the
    # single source of truth for which channels run. For a wrist watch
    # (IBI+HR+SKIN_TEMP) this yields [hrv, hr, (temp when it has data)]. No
    # declaration means no channels: the pipeline already skips decide() in that
    # case, and an empty set here keeps the two consistent (no hidden default).
    if present is None:
        present = frozenset()
    channels = build_channels(ChannelContext(
        primary=primary, sensor=sensor, quality=quality, baseline=baseline,
        temperature=temperature, present=present,
    ))
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
