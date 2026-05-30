"""The capability contract: presence flags + capability subset checks."""

from __future__ import annotations

from affectus.sensing.capabilities import (
    Capability,
    DeviceCapabilities,
    SensorFrame,
)


def test_has_and_has_all():
    caps = DeviceCapabilities(
        present=frozenset({Capability.IBI, Capability.HR, Capability.PPG}),
        profile="wrist_ppg",
    )
    assert caps.has(Capability.IBI)
    assert not caps.has(Capability.EDA)
    assert caps.has_all(frozenset({Capability.IBI, Capability.HR}))
    assert not caps.has_all(frozenset({Capability.IBI, Capability.EDA}))


def test_frame_absent_slots_are_none():
    caps = DeviceCapabilities(present=frozenset({Capability.IBI}))
    frame = SensorFrame(timestamp_ms=1000, capabilities=caps)
    assert frame.has(Capability.IBI)
    assert not frame.has(Capability.SKIN_TEMP)
    # absent capability -> slot stays None
    assert frame.skin_temp_c is None
    assert frame.ppg is None


def test_profile_defaults_to_wrist():
    caps = DeviceCapabilities(present=frozenset())
    assert caps.profile == "wrist_ppg"
