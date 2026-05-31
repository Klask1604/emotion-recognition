"""
WRIST device family (Galaxy Watch optical-PPG wearable).

Declares everything specific to a wrist device in one place:
  - profile:  the wrist_ppg threshold profile (registered on import)
  - adapter:  raw schema-v2 batch -> SensorFrame
  - modules:  the per-sensor feature-modules this family can run

Each module declares the Capability it REQUIRES; devices.registry builds only the
modules whose capability is present in a given frame. So a wrist device that
happens not to send PPG simply never builds the valence module — no branching.

The generic algorithms (HRV math, signal quality, baseline, fusion) are NOT here;
they live in shared/ and are reused by every family.
"""

from __future__ import annotations

from affectus.devices.wrist import profile as _profile  # noqa: F401 (registers wrist_ppg)
from affectus.devices.wrist.adapter import frame_from_acquisition_v2  # noqa: F401
from affectus.devices.wrist.modules import hr, hrv, temperature, valence_wesad

PROFILE = "wrist_ppg"

# The feature-modules available to a wrist device, each declaring its REQUIRES.
# Order is diagnostic only; fusion is order-independent. valence_wesad is
# default-disabled (cap 0) so it is an exact no-op until enabled.
MODULES = [hrv, hr, temperature, valence_wesad]

__all__ = ["PROFILE", "MODULES", "frame_from_acquisition_v2"]
