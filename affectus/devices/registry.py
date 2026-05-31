"""
Device-module registry: capability-gated assembly of the fusion modules.

Every device family (devices/<family>) exposes a MODULES list; each module
declares the Capabilities it REQUIRES and an evaluate(ctx) -> FusionChannel|None.
This registry collects all modules and, for a given frame/context, builds only
those whose required capabilities are present — and which return a non-None
contribution. Two clean gates, kept separate:
  - capability present?  -> run the module at all
  - weight > 0?          -> the module actually contributes (existing math)

For a wrist frame (IBI + HR + SKIN_TEMP) this yields exactly [hrv, hr, temp], so
the fused verdict is unchanged. A device without SKIN_TEMP simply never builds
the temp module (it is skipped, not a weight-0 channel).
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.contract.capabilities import Capability, DeviceCapabilities
from affectus.shared.fusion import FusionChannel


@dataclass
class ChannelContext:
    """Everything a module evaluator may need for one epoch. Assembled by the
    pipeline; modules read only what their capability grants."""

    primary: object                      # HrvMetrics
    sensor: object | None                # SensorBatchMessage
    quality: object                      # SignalQuality
    baseline: object                     # RestBaselineStore
    temperature: object | None = None    # SkinTemperatureChannelState
    valence_wesad: object | None = None  # ValenceWesadState
    ppg_green: list | None = None        # raw PPG window (for valence features)
    ppg_ts_ms: list | None = None
    present: frozenset[Capability] = frozenset()


def _all_modules() -> list:
    """Gather the feature-modules from every device family. Imported lazily so
    this module has no import-time dependency on the device packages."""
    import affectus.devices as devices

    modules: list = []
    for family in (devices.wrist, devices.chest, devices.head):
        modules.extend(getattr(family, "MODULES", []))
    return modules


def build_channels(ctx: ChannelContext) -> list[FusionChannel]:
    """Run every registered module whose required capabilities are present and
    that returns a non-None contribution."""
    out: list[FusionChannel] = []
    for module in _all_modules():
        if not (module.REQUIRES <= ctx.present):
            continue
        ch = module.evaluate(ctx)
        if ch is not None:
            out.append(ch)
    return out


def modules_for(caps: DeviceCapabilities) -> list[str]:
    """Names of the feature-modules that WOULD run for a device with these
    capabilities (capability gate only — independent of runtime weight). Used by
    the handshake ack so the watch is told which modules it activates. Modules
    flagged ANNOUNCED=False (default-disabled / not production-trusted) are
    excluded so the ack reflects what genuinely contributes."""
    return [
        module.NAME
        for module in _all_modules()
        if module.REQUIRES <= caps.present
        and getattr(module, "ANNOUNCED", True)
    ]
