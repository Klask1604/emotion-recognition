"""Single decision gate: Kubios physiology + HAR caps + alert confirmation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from biofizic.config import (
    ALERT_CONFIRMATION_EPOCH_COUNT,
    ALERT_PENDING_CAP,
    HAR_AROUSAL_CAP_BY_CLASS,
    REST_ACCELERATION_P90_MAX,
    RMSSD_SPIKE_RATIO_THRESHOLD,
    RMSSD_Z_SUPPRESS_ALERT,
    STRESS_INDEX_Z_ALERT,
    STRESS_INDEX_Z_ALERT_STRONG,
)
from biofizic.decision.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    stress_index_to_arousal,
    zone_is_alert_or_higher,
    zone_is_elevated_or_higher,
)
from biofizic.motion.motion_ml import MotionPrediction


@dataclass
class DecisionGateState:
    """Mutable state across epochs — one elevated_streak counter."""

    elevated_streak: int = 0
    recent_rmssd: deque[float] = field(default_factory=lambda: deque(maxlen=4))


@dataclass(frozen=True)
class DecisionGateResult:
    display_arousal_10: int
    display_label: str
    kubios_label: str
    decision_reason: str
    gate_mode: str


def _is_rest(motion_class: str, acc_p90: float) -> bool:
    """Rest-like when HAR says STILL and wrist acceleration is low."""
    return motion_class == "STILL" and acc_p90 <= REST_ACCELERATION_P90_MAX


def _dual_ok(stress_index_z: float, rmssd_z: float) -> bool:
    return (
        stress_index_z >= STRESS_INDEX_Z_ALERT
        and rmssd_z >= RMSSD_Z_SUPPRESS_ALERT
    ) or stress_index_z >= STRESS_INDEX_Z_ALERT_STRONG


def apply_decision_gate(
    *,
    kubios_stress_index: float,
    rmssd_ms: float,
    stress_index_z: float,
    rmssd_z: float,
    motion: MotionPrediction,
    acc_p90: float,
    gate_state: DecisionGateState,
) -> DecisionGateResult:
    """
    Fuse Kubios stress index with HAR motion caps and confirm alerts.

    Order: physio scale -> HAR cap -> RMSSD spike filter -> rest dual-veto
    -> elevated streak -> alert confirmation (ALERT_CONFIRMATION_EPOCH_COUNT).
    """
    if kubios_stress_index <= 0:
        gate_state.elevated_streak = 0
        return DecisionGateResult(
            display_arousal_10=5,
            display_label="Echilibrat",
            kubios_label="Echilibrat",
            decision_reason="no_stress_index",
            gate_mode="kubios_zone",
        )

    zone = kubios_zone_for_stress_index(kubios_stress_index)
    _, physio_scale_10, _ = stress_index_to_arousal(kubios_stress_index)
    kubios_label = zone.label
    cap = HAR_AROUSAL_CAP_BY_CLASS.get(motion.motion_class, ALERT_PENDING_CAP)
    display_a10 = min(int(physio_scale_10), cap)

    reasons = [
        f"physio={kubios_label}",
        f"har={motion.motion_class}",
        f"har_conf={motion.confidence:.2f}",
    ]
    if display_a10 < int(physio_scale_10):
        reasons.append(f"cap_{cap}")

    gate_state.recent_rmssd.append(rmssd_ms)

    if (
        len(gate_state.recent_rmssd) >= 2
        and _is_rest(motion.motion_class, acc_p90)
        and rmssd_ms > 0
    ):
        previous = list(gate_state.recent_rmssd)[:-1]
        median = float(np.median(previous)) if previous else rmssd_ms
        if median > 0 and abs(rmssd_ms - median) / median > RMSSD_SPIKE_RATIO_THRESHOLD:
            gate_state.elevated_streak = 0
            capped = min(display_a10, ALERT_PENDING_CAP)
            return DecisionGateResult(
                display_arousal_10=capped,
                display_label=arousal_scale_10_to_label(capped),
                kubios_label=kubios_label,
                decision_reason="|".join(reasons + ["rmssd_spike_cap"]),
                gate_mode="rmssd_spike_cap",
            )

    if not zone_is_elevated_or_higher(zone):
        gate_state.elevated_streak = 0
        return DecisionGateResult(
            display_arousal_10=display_a10,
            display_label=arousal_scale_10_to_label(display_a10),
            kubios_label=kubios_label,
            decision_reason="|".join(reasons),
            gate_mode="kubios_zone",
        )

    if _is_rest(motion.motion_class, acc_p90) and not _dual_ok(stress_index_z, rmssd_z):
        gate_state.elevated_streak = 0
        capped = min(display_a10, ALERT_PENDING_CAP)
        return DecisionGateResult(
            display_arousal_10=capped,
            display_label=arousal_scale_10_to_label(capped),
            kubios_label=kubios_label,
            decision_reason="|".join(reasons + ["rest_dual_veto"]),
            gate_mode="rest_dual_veto",
        )

    gate_state.elevated_streak += 1
    gate_mode = "kubios_zone"

    if zone_is_alert_or_higher(zone):
        if gate_state.elevated_streak < ALERT_CONFIRMATION_EPOCH_COUNT:
            capped = min(display_a10, ALERT_PENDING_CAP)
            return DecisionGateResult(
                display_arousal_10=capped,
                display_label=arousal_scale_10_to_label(capped),
                kubios_label=kubios_label,
                decision_reason="|".join(reasons + ["alert_pending"]),
                gate_mode="alert_pending",
            )
        gate_mode = "alert_confirmed"

    return DecisionGateResult(
        display_arousal_10=display_a10,
        display_label=arousal_scale_10_to_label(display_a10),
        kubios_label=kubios_label,
        decision_reason="|".join(reasons),
        gate_mode=gate_mode,
    )
