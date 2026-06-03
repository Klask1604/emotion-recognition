"""
SensorProfile: device-modality-dependent thresholds, shared infrastructure.

The signal-quality and baseline algorithms are written once (shared/); the few
thresholds that genuinely depend on the sensor MODALITY (not the subject) live in
a SensorProfile, keyed by a profile name carried on the frame's
DeviceCapabilities. The shared algorithm reads the right number from the profile
— it never branches on device.

Each device registers its own profile via register_profile() from its profile
module (today: contract/wrist_profile.py). This module owns only the structure +
lookup; the values live with the device that owns them.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SensorProfile:
    """Device-modality-dependent thresholds. Add fields here only when a value
    truly differs by sensor type, not by individual subject."""

    # Max IBI artifact rate before an epoch is untrusted. Consumer wrist PPG runs
    # 10-20% corrected beats even on a good recording, so the cutoff is loose
    # (0.15); a chest ECG would use ~0.05 because R-peaks are clean.
    artifact_rate_max: float
    # When the optical sensor is corrupted by motion, multiply HRV quality by
    # this. Wrist PPG is very motion-fragile; a chest strap far less so.
    motion_moving_quality_factor: float


_PROFILES: dict[str, SensorProfile] = {}
_DEFAULT = "wrist_ppg"


def register_profile(name: str, profile: SensorProfile) -> None:
    """Called once per device family (at import) to register its thresholds."""
    _PROFILES[name] = profile


def profile_for(name: str) -> SensorProfile:
    """Resolve a profile by name, falling back to wrist_ppg for unknown devices.

    Device profiles are imported lazily here so registration happens before the
    first lookup without creating an import cycle (shared/ must not import the
    profile modules at module load)."""
    if not _PROFILES:
        import affectus.contract.wrist_profile  # noqa: F401  (registers wrist_ppg)
    return _PROFILES.get(name) or _PROFILES[_DEFAULT]
