"""
The server-defined handshake contract.

The watch ANNOUNCES which sensors it carries on `biofizic/hello`; the server
VALIDATES that announcement and replies on `biofizic/hello/ack` with OK + the
modules that will be active, or an error if the device cannot be classified
(e.g. it offers only skin temperature). The watch waits for OK before streaming.

This module owns: parsing the announced capability list, the minimum-capability
rule for classifiability, building the ack payload, and the present-capabilities ->
active-channels mapping (modules_for). That mapping mirrors
engine/channels.build_channels one gate per signal, so the ack never drifts from
what the pipeline actually runs.
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.contract.capabilities import Capability, DeviceCapabilities

# A device must carry at least one of these for arousal classification to be
# possible at all. IBI drives HRV; PPG can be peak-detected into IBI. Skin temp
# or HR alone cannot anchor a verdict.
CLASSIFIABLE_ANY: frozenset[Capability] = frozenset({Capability.IBI, Capability.PPG})


@dataclass(frozen=True)
class Ack:
    status: str                     # "ok" | "error"
    profile: str                    # threshold profile the server will apply
    modules_active: list[str]       # feature-modules that will run for this device
    min_required: list[str]         # human-readable minimum to be classifiable
    reason: str = ""                # populated on error

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "profile": self.profile,
            "modules_active": self.modules_active,
            "min_required": self.min_required,
            "reason": self.reason,
        }


def parse_capabilities(names: list[str]) -> frozenset[Capability]:
    """Map announced capability strings to the Capability enum, ignoring unknown
    ones (forward-compatible: a newer watch may announce capabilities this server
    doesn't model yet)."""
    out = set()
    by_value = {c.value: c for c in Capability}
    for n in names or []:
        c = by_value.get(str(n).strip().lower())
        if c is not None:
            out.add(c)
    return frozenset(out)


def validate_announcement(
    capability_names: list[str], profile: str = "wrist_ppg"
) -> Ack:
    """Validate a device's announced capabilities and build the ack.

    Returns an error ack when the device offers nothing classifiable (no IBI and
    no PPG); otherwise an ok ack listing the modules that will be active for the
    present capability set."""
    present = parse_capabilities(capability_names)
    min_required = ["ibi or ppg"]

    if not (present & CLASSIFIABLE_ANY):
        return Ack(
            status="error",
            profile=profile,
            modules_active=[],
            min_required=min_required,
            reason="insufficient sensors to classify arousal: need IBI or PPG "
                   f"(announced: {sorted(c.value for c in present)})",
        )

    caps = DeviceCapabilities(present=present, profile=profile)
    return Ack(
        status="ok",
        profile=profile,
        modules_active=modules_for(caps),
        min_required=min_required,
    )


# Which fusion channels a device with these capabilities activates. Mirrors
# engine/channels.build_channels exactly (one gate per signal) so the handshake
# ack never drifts from what the pipeline runs. valence_wesad is research-only and
# not announced, so it is intentionally absent here.
_ANNOUNCED_CHANNELS: list[tuple[str, Capability]] = [
    ("hrv", Capability.IBI),
    ("hr", Capability.HR),
    ("temp", Capability.SKIN_TEMP),
]


def modules_for(caps: DeviceCapabilities) -> list[str]:
    """Names of the fusion channels that WOULD run for a device with these
    capabilities (capability gate only, independent of runtime weight). Used by
    the handshake ack so the watch is told which channels it activates."""
    return [name for name, cap in _ANNOUNCED_CHANNELS if cap in caps.present]
