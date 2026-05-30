"""Per-capability profiles: the wrist default reproduces prior behaviour, and a
different profile changes the quality scoring (proving thresholds are honoured)."""

from __future__ import annotations

from affectus.engine.signal_quality import SignalQualityState, update_and_score
from affectus.sensing.profiles import SensorProfile, profile_for


def test_default_profile_is_wrist():
    assert profile_for("wrist_ppg").artifact_rate_max == 0.15
    # unknown device falls back to wrist
    assert profile_for("nonexistent").artifact_rate_max == 0.15


def test_stricter_profile_lowers_quality_at_same_artifact_rate():
    # Same artifact rate, two profiles: the stricter cutoff yields lower quality.
    wrist = profile_for("wrist_ppg")                       # artifact_max 0.15
    strict = SensorProfile(artifact_rate_max=0.05, motion_moving_quality_factor=0.1)

    def quality(profile):
        st = SignalQualityState()
        # prime the motion baseline as "still"
        for _ in range(10):
            update_and_score(motion_energy=0.0, artifact_rate=0.0, state=st, profile=profile)
        return update_and_score(
            motion_energy=0.0, artifact_rate=0.08, state=st, profile=profile
        ).quality

    q_wrist = quality(wrist)
    q_strict = quality(strict)
    assert q_strict < q_wrist   # 0.08 is fine for wrist (0.15), poor for strict (0.05)


def test_omitting_profile_equals_wrist_profile():
    def q(profile):
        st = SignalQualityState()
        for _ in range(10):
            update_and_score(motion_energy=0.0, artifact_rate=0.0, state=st, profile=profile)
        return update_and_score(
            motion_energy=0.0, artifact_rate=0.1, state=st, profile=profile
        ).quality

    assert q(None) == q(profile_for("wrist_ppg"))
