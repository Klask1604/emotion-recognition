"""
Backward-compatible re-exports. Canonical: biofizic.constants.kubios_zones and decision.arousal_mapper
"""

from biofizic.constants.kubios_zones import (
    ALERT_CONFIRMATION_EPOCH_COUNT as ALERT_CONFIRM_EPOCHS,
    BAEVSKY_RAW_NORMAL_HIGH as BAEVSKY_NORMAL_RAW_HI,
    BAEVSKY_RAW_NORMAL_LOW as BAEVSKY_NORMAL_RAW_LO,
    BASELINE_EMA_ALPHA as PASSIVE_EMA_ALPHA,
    ML_MIN_ACCURACY as ML_GATE_ACCURACY_MIN,
    ML_MIN_F1 as ML_GATE_F1_MIN,
    MOTION_AROUSAL_CAP,
    REST_ACCELERATION_P90_MAX as REST_ACC_P90_MAX,
    RMSSD_SPIKE_RATIO_THRESHOLD as RMSSD_SPIKE_RATIO,
    RMSSD_Z_SUPPRESS_ALERT as Z_RMSSD_SUPPRESS,
    STILL_EPOCHS_BEFORE_BASELINE_LOCK as PASSIVE_REST_EPOCHS_MIN,
    STRESS_INDEX_Z_ALERT as Z_SI_ALERT,
    STRESS_INDEX_Z_ALERT_STRONG as Z_SI_ALERT_STRONG,
    KubiosZoneId as KubiosSiZone,
)
from biofizic.decision.arousal_mapper import (
    KubiosZone as ZoneInfo,
    arousal_scale_10_to_label as arousal_10_to_label,
    baseline_z_score_to_label as z_si_to_label,
    kubios_zone_for_stress_index as kubios_zone,
    stress_index_to_arousal,
    zone_is_alert_or_higher,
    zone_is_elevated_or_higher,
)

# Legacy population constants
RMSSD_POP_P5_MS = 19.0
RMSSD_POP_P95_MS = 75.0
RMSSD_POP_MEAN_MS = 42.0
BAEVSKY_MILD_STRESS_RATIO = 1.5
BAEVSKY_SEVERE_STRESS_RATIO = 5.0
PROFILE_ACC_TUKEY_K = 1.5
ML_TIEBREAKER_W_V3_MAX = 0.25
ML_TIEBREAKER_CONF_MIN = 0.65

SI_ZONE_BOUNDS = (
    (7.1, KubiosSiZone.LOW, "Relaxat", "low", 0.20, 2),
    (12.2, KubiosSiZone.NORMAL, "Echilibrat", "normal", 0.50, 5),
    (22.4, KubiosSiZone.ELEVATED, "Moderat", "elevated", 0.68, 7),
    (30.0, KubiosSiZone.HIGH, "Alert", "high", 0.82, 8),
    (float("inf"), KubiosSiZone.VERY_HIGH, "Ridicat", "very_high", 0.95, 10),
)


def clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def hr_to_arousal_10(mean_hr: float, rest_hr: float | None = None) -> int:
    if mean_hr <= 0:
        return 5
    base = rest_hr if rest_hr and rest_hr > 0 else 70.0
    z = clip((mean_hr - base) / max(8.0, base * 0.12), -2.0, 3.0)
    if z < -0.5:
        return 2
    if z < 0.3:
        return 4
    if z < 0.8:
        return 5
    if z < 1.3:
        return 6
    if z < 1.8:
        return 7
    return 8


def hr_to_label(mean_hr: float, rest_hr: float | None = None) -> str:
    return arousal_10_to_label(hr_to_arousal_10(mean_hr, rest_hr))
