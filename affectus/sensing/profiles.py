"""
Per-capability threshold profiles.

The signal-quality and (later) baseline algorithms are written once and shared;
the few thresholds that genuinely depend on the sensor *modality* live here,
keyed by a profile name carried on the frame's DeviceCapabilities. The shared
algorithm reads the right number from the profile — it never branches on device.

Today only 'wrist_ppg' is populated (reproducing the current config values, so
behaviour is byte-for-byte unchanged). 'chest_ecg' is a future slot.
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.config import ARTIFACT_RATE_MAX, MOTION_MOVING_QUALITY_FACTOR


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


PROFILES: dict[str, SensorProfile] = {
    "wrist_ppg": SensorProfile(
        artifact_rate_max=ARTIFACT_RATE_MAX,                 # 0.15 today
        motion_moving_quality_factor=MOTION_MOVING_QUALITY_FACTOR,  # 0.1 today
    ),
    # Future slot — chest ECG has clean R-peaks and little optical-motion coupling:
    # "chest_ecg": SensorProfile(artifact_rate_max=0.05, motion_moving_quality_factor=0.5),
}


def profile_for(name: str) -> SensorProfile:
    """Resolve a profile by name, falling back to wrist_ppg for unknown devices."""
    return PROFILES.get(name, PROFILES["wrist_ppg"])
