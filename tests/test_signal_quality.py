"""Signal-quality gate: artifact cutoff, quality monotonicity, learned still/moving."""

from __future__ import annotations

from biofizic.config import ARTIFACT_RATE_MAX
from biofizic.engine.signal_quality import (
    SignalQualityState,
    update_and_score,
)


def test_epoch_unusable_when_artifact_rate_above_cutoff():
    state = SignalQualityState()
    good = update_and_score(motion_energy=0.0, artifact_rate=0.0, state=state)
    bad = update_and_score(
        motion_energy=0.0, artifact_rate=ARTIFACT_RATE_MAX + 0.2, state=state
    )
    assert good.usable
    assert not bad.usable


def test_quality_drops_as_motion_energy_rises():
    # Train the per-user model so motion predicts artifacts, then compare Q at
    # low vs high motion energy at the same (clean) artifact rate.
    state = SignalQualityState()
    for _ in range(50):
        update_and_score(motion_energy=0.0, artifact_rate=0.0, state=state)
        update_and_score(motion_energy=5.0, artifact_rate=0.5, state=state)
    low = update_and_score(motion_energy=0.0, artifact_rate=0.0, state=state)
    high = update_and_score(motion_energy=5.0, artifact_rate=0.0, state=state)
    assert low.quality > high.quality
    assert 0.0 <= high.quality <= 1.0 and 0.0 <= low.quality <= 1.0


def test_motion_state_from_energy_not_artifacts():
    # High artifact rate at near-zero motion (noisy wrist PPG at rest) must NOT
    # be read as movement.
    state = SignalQualityState()
    for _ in range(30):
        r = update_and_score(motion_energy=0.0, artifact_rate=0.3, state=state)
    assert r.motion_state == "still"
    # A clear upper outlier in motion energy is movement.
    assert update_and_score(motion_energy=2.0, artifact_rate=0.0, state=state).motion_state == "moving"


def test_motion_state_still_during_cold_start():
    state = SignalQualityState()
    # Before MIN_QUALITY_UPDATES samples we have no baseline -> assume still.
    assert update_and_score(motion_energy=0.0, artifact_rate=0.0, state=state).motion_state == "still"
