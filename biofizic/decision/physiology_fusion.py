"""Fuse HAR motion class with Kubios physiology into display output."""

from __future__ import annotations

from biofizic.constants.kubios_zones import STRESS_INDEX_Z_ALERT, STRESS_INDEX_Z_ALERT_STRONG
from biofizic.constants.motion import HAR_AROUSAL_CAP_BY_CLASS
from biofizic.decision.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    stress_index_to_arousal,
)
from biofizic.motion.motion_ml import MotionPrediction


def fuse_physiology_and_motion(
    *,
    kubios_stress_index: float,
    motion: MotionPrediction,
    stress_index_z_score: float,
    rmssd_z_score: float,
    elevated_streak: int,
) -> tuple[int, str, str, str]:
    if kubios_stress_index <= 0:
        return 5, "Echilibrat", "Echilibrat", "no_stress_index"

    zone = kubios_zone_for_stress_index(kubios_stress_index)
    _, physio_scale_10, _ = stress_index_to_arousal(kubios_stress_index)
    kubios_label = zone.label
    cap = HAR_AROUSAL_CAP_BY_CLASS.get(motion.motion_class, 6)
    display_scale_10 = min(int(physio_scale_10), cap)

    reasons = [
        f"physio={kubios_label}",
        f"har={motion.motion_class}",
        f"har_conf={motion.confidence:.2f}",
    ]

    if display_scale_10 < physio_scale_10:
        reasons.append(f"cap_{cap}")

    if motion.motion_class == "STILL" and zone.band_id in ("high", "very_high"):
        dual_ok = (
            stress_index_z_score >= STRESS_INDEX_Z_ALERT
            and rmssd_z_score >= -1.0
        ) or stress_index_z_score >= STRESS_INDEX_Z_ALERT_STRONG
        if elevated_streak < 2 or not dual_ok:
            display_scale_10 = min(display_scale_10, 6)
            reasons.append("alert_pending")

    display_label = arousal_scale_10_to_label(display_scale_10)
    return display_scale_10, display_label, kubios_label, "|".join(reasons)
