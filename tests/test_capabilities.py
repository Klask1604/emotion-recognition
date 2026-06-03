"""The capability contract: presence flags + capability subset checks."""

from __future__ import annotations

from affectus.contract.capabilities import Capability, DeviceCapabilities


def test_has_and_has_all():
    caps = DeviceCapabilities(
        present=frozenset({Capability.IBI, Capability.HR, Capability.PPG}),
        profile="wrist_ppg",
    )
    assert caps.has(Capability.IBI)
    assert not caps.has(Capability.EDA)
    assert caps.has_all(frozenset({Capability.IBI, Capability.HR}))
    assert not caps.has_all(frozenset({Capability.IBI, Capability.EDA}))


def test_profile_defaults_to_wrist():
    caps = DeviceCapabilities(present=frozenset())
    assert caps.profile == "wrist_ppg"
