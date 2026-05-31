"""Device-module registry + handshake validation: modules build only when their
required capabilities are present; the ack reflects the same gating; devices that
cannot be classified are rejected."""

from __future__ import annotations

from affectus.contract.capabilities import Capability, DeviceCapabilities
from affectus.contract.schema import validate_announcement
from affectus.devices.registry import modules_for


def _caps(*present, profile="wrist_ppg"):
    return DeviceCapabilities(frozenset(present), profile=profile)


def test_wrist_full_modules():
    # IBI + HR + SKIN_TEMP -> hrv, hr, temp (valence excluded: ANNOUNCED=False).
    mods = modules_for(_caps(Capability.IBI, Capability.HR, Capability.SKIN_TEMP))
    assert "hrv" in mods and "hr" in mods and "temp" in mods
    assert "valence_wesad" not in mods


def test_no_skin_temp_skips_temp_module():
    mods = modules_for(_caps(Capability.IBI, Capability.HR))
    assert "temp" not in mods
    assert {"hrv", "hr"} <= set(mods)


def test_no_hr_skips_hr_module():
    mods = modules_for(_caps(Capability.IBI))
    assert "hr" not in mods
    assert mods == ["hrv"]


def test_announcement_ok_lists_modules():
    ack = validate_announcement(["ibi", "hr", "ppg", "motion", "temp"])
    assert ack.status == "ok"
    assert ack.modules_active == ["hrv", "hr", "temp"]


def test_announcement_skin_temp_only_is_error():
    ack = validate_announcement(["temp"])
    assert ack.status == "error"
    assert "ibi or ppg" in ack.reason.lower() or "ibi" in ack.reason.lower()
    assert ack.modules_active == []


def test_announcement_ignores_unknown_capabilities():
    # A forward-compatible watch may announce something this server doesn't model;
    # it must not crash, and IBI still makes it classifiable.
    ack = validate_announcement(["ibi", "made_up_sensor"])
    assert ack.status == "ok"


def test_chest_head_families_are_empty_templates():
    import affectus.devices as devices
    assert devices.chest.MODULES == []
    assert devices.head.MODULES == []
