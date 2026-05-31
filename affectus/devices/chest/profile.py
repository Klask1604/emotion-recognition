"""Chest ECG threshold profile — TEMPLATE (not registered yet).

When the chest family is implemented, uncomment the registration. R-peaks are
clean so the artifact cutoff is tight (0.05) and the strap is far less
motion-coupled than wrist optics (motion factor 0.5)."""

from __future__ import annotations

# from affectus.shared.profiles import SensorProfile, register_profile
#
# CHEST_ECG_PROFILE = SensorProfile(
#     artifact_rate_max=0.05,
#     motion_moving_quality_factor=0.5,
# )
# register_profile("chest_ecg", CHEST_ECG_PROFILE)
