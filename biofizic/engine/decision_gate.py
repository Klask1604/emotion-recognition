"""Decision gate: arousal from the Kalman-filtered personal z + CUSUM alert.

The signal-quality gate no longer holds a value by hand: the upstream Kalman
smoother (engine/state_estimator.py) is fed a measurement variance that grows
when quality is poor, so low-quality epochs barely move the estimate. This gate
just maps the filtered z to a displayed arousal and runs the CUSUM change
detector:
  - personal arousal = Phi(z_filtered) once the baseline is locked, falling back
    to the population Kubios zone before that,
  - a one-sided CUSUM change detector on z_filtered for sustained-stress alerts.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from biofizic.engine.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    personal_arousal_10,
    population_arousal_10,
)
from biofizic.engine.cusum import CusumDetector
from biofizic.engine.signal_quality import SignalQuality


@dataclass
class DecisionGateState:
    """Mutable state across epochs."""

    cusum: CusumDetector = field(default_factory=CusumDetector)


@dataclass(frozen=True)
class DecisionGateResult:
    display_arousal_10: int
    display_label: str
    kubios_label: str
    confidence: float
    alert: bool
    decision_reason: str
    gate_mode: str


def apply_decision_gate(
    *,
    kubios_stress_index: float,
    stress_index_z_filtered: float,
    quality: SignalQuality,
    baseline_ready: bool,
    gate_state: DecisionGateState,
    arousal_offset_z: float = 0.0,
) -> DecisionGateResult:
    kubios_label = kubios_zone_for_stress_index(kubios_stress_index).label

    if baseline_ready:
        arousal_10 = personal_arousal_10(stress_index_z_filtered, arousal_offset_z)
        gate_mode = "personal_z"
    else:
        arousal_10 = population_arousal_10(kubios_stress_index)
        gate_mode = "population_zone"
    display_label = arousal_scale_10_to_label(arousal_10)

    reasons = [
        f"kubios={kubios_label}",
        f"motion={quality.motion_state}",
        f"q={quality.quality:.2f}",
        f"artifact={quality.artifact_rate:.2f}",
    ]

    # CUSUM on the filtered z (meaningful only once the baseline is locked). The
    # filtered z is already quality-attenuated, so artifact bursts cannot push it.
    alert = gate_state.cusum.update(stress_index_z_filtered) if baseline_ready else False
    if alert:
        gate_mode = "alert_confirmed"
        reasons.append("alert")

    return DecisionGateResult(
        display_arousal_10=arousal_10,
        display_label=display_label,
        kubios_label=kubios_label,
        confidence=quality.quality,
        alert=alert,
        decision_reason="|".join(reasons),
        gate_mode=gate_mode,
    )
