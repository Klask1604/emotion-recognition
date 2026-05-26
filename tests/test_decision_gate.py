"""Single-tick gate logic: HAR cap, rest dual-veto, alert confirmation, spike cap."""

from __future__ import annotations

from biofizic.config import ALERT_CONFIRMATION_EPOCH_COUNT, ALERT_PENDING_CAP
from biofizic.decision.decision_gate import (
    DecisionGateState,
    apply_decision_gate,
)
from biofizic.motion.motion_ml import MotionPrediction


def _motion(motion_class: str, confidence: float = 0.9) -> MotionPrediction:
    return MotionPrediction(
        motion_class=motion_class,
        confidence=confidence,
        probabilities={motion_class: confidence},
    )


def test_zero_stress_index_returns_neutral():
    state = DecisionGateState()
    result = apply_decision_gate(
        kubios_stress_index=0.0,
        rmssd_ms=0.0,
        stress_index_z=0.0,
        rmssd_z=0.0,
        motion=_motion("STILL"),
        acc_p90=0.0,
        gate_state=state,
    )
    assert result.display_arousal_10 == 5
    assert result.display_label == "Echilibrat"
    assert result.decision_reason == "no_stress_index"


def test_walk_caps_arousal():
    state = DecisionGateState()
    # SI well into the alert zone but the user is WALK, so display must be capped.
    result = apply_decision_gate(
        kubios_stress_index=40.0,
        rmssd_ms=20.0,
        stress_index_z=2.0,
        rmssd_z=0.0,
        motion=_motion("WALK"),
        acc_p90=1.5,
        gate_state=state,
    )
    assert result.display_arousal_10 <= 5
    assert "har=WALK" in result.decision_reason


def test_alert_requires_consecutive_confirmation():
    state = DecisionGateState()
    # First HIGH epoch should be capped at ALERT_PENDING_CAP and marked pending.
    first = apply_decision_gate(
        kubios_stress_index=27.0,  # HIGH zone
        rmssd_ms=20.0,
        stress_index_z=1.6,
        rmssd_z=0.0,
        motion=_motion("STILL"),
        acc_p90=0.1,
        gate_state=state,
    )
    assert first.display_arousal_10 == ALERT_PENDING_CAP
    assert first.gate_mode == "alert_pending"

    # Second consecutive HIGH epoch is enough to confirm.
    assert ALERT_CONFIRMATION_EPOCH_COUNT == 2
    second = apply_decision_gate(
        kubios_stress_index=27.0,
        rmssd_ms=20.0,
        stress_index_z=1.6,
        rmssd_z=0.0,
        motion=_motion("STILL"),
        acc_p90=0.1,
        gate_state=state,
    )
    assert second.display_arousal_10 > ALERT_PENDING_CAP
    assert second.gate_mode == "alert_confirmed"


def test_rest_dual_veto_caps_when_rmssd_does_not_agree():
    state = DecisionGateState()
    # ELEVATED SI but no RMSSD confirmation -> rest dual-veto path.
    result = apply_decision_gate(
        kubios_stress_index=15.0,  # ELEVATED zone
        rmssd_ms=60.0,
        stress_index_z=0.5,  # below STRESS_INDEX_Z_ALERT
        rmssd_z=0.0,
        motion=_motion("STILL"),
        acc_p90=0.1,
        gate_state=state,
    )
    assert result.gate_mode == "rest_dual_veto"
    assert result.display_arousal_10 <= ALERT_PENDING_CAP
