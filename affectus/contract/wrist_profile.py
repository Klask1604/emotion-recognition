"""Wrist (optical PPG) threshold profile. Reproduces the current config values
exactly so behaviour is byte-for-byte unchanged."""

from __future__ import annotations

from affectus.config import ARTIFACT_RATE_MAX, MOTION_MOVING_QUALITY_FACTOR
from affectus.dsp.profiles import SensorProfile, register_profile

WRIST_PPG_PROFILE = SensorProfile(
    artifact_rate_max=ARTIFACT_RATE_MAX,                        # 0.15 today
    motion_moving_quality_factor=MOTION_MOVING_QUALITY_FACTOR,  # 0.1 today
)

register_profile("wrist_ppg", WRIST_PPG_PROFILE)
