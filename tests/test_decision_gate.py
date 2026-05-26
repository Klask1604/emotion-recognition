"""Decision gate: population vs personal arousal from filtered z, CUSUM alert."""

from __future__ import annotations

from biofizic.config import CUSUM_THRESHOLD_H
from biofizic.engine.decision_gate import DecisionGateState, apply_decision_gate
from biofizic.engine.signal_quality import SignalQuality


def _quality(*, q: float = 0.9, motion: str = "still", artifact: float = 0.0) -> SignalQuality:
    return SignalQuality(
        quality=q,
        usable=artifact <= 0.05,
        artifact_rate=artifact,
        motion_energy=0.0,
        p_artifact=1.0 - q,
        motion_state=motion,
    )


def test_pre_baseline_uses_population_zone():
    state = DecisionGateState()
    result = apply_decision_gate(
        kubios_stress_index=10.0,  # NORMAL zone -> arousal 5
        stress_index_z_filtered=0.0,
        quality=_quality(),
        baseline_ready=False,
        gate_state=state,
    )
    assert result.gate_mode == "population_zone"
    assert result.display_arousal_10 == 5
    assert result.kubios_label == "Echilibrat"


def test_post_baseline_uses_filtered_z():
    state = DecisionGateState()
    high = apply_decision_gate(
        kubios_stress_index=10.0,
        stress_index_z_filtered=3.0,  # far above personal baseline -> high arousal
        quality=_quality(),
        baseline_ready=True,
        gate_state=state,
    )
    assert high.gate_mode in ("personal_z", "alert_confirmed")
    assert high.display_arousal_10 == 10


def test_cusum_confirms_sustained_elevation():
    state = DecisionGateState()
    fired = any(
        apply_decision_gate(
            kubios_stress_index=25.0,
            stress_index_z_filtered=2.0,
            quality=_quality(),
            baseline_ready=True,
            gate_state=state,
        ).alert
        for _ in range(20)
    )
    assert fired
    assert state.cusum.s_hi > CUSUM_THRESHOLD_H


def test_cusum_quiet_under_noise():
    state = DecisionGateState()
    for i in range(50):
        z = 0.3 if i % 2 == 0 else -0.3
        r = apply_decision_gate(
            kubios_stress_index=10.0,
            stress_index_z_filtered=z,
            quality=_quality(),
            baseline_ready=True,
            gate_state=state,
        )
        assert not r.alert
