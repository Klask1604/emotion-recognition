"""
Kubios stress index zones and alert confirmation thresholds.

stress_index in this project means sqrt(Baevsky SI) as defined in Kubios HRV User Guide 3.x.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# Passive baseline: minimum STILL epochs before lock-in.
STILL_EPOCHS_BEFORE_BASELINE_LOCK = 4

# Exponential moving average weight for baseline updates during STILL only.
BASELINE_EMA_ALPHA = 0.05

# Alert confirmation: consecutive elevated epochs required.
ALERT_CONFIRMATION_EPOCH_COUNT = 2

# Z-score thresholds for alert dual-gate at rest.
STRESS_INDEX_Z_ALERT = 1.0
STRESS_INDEX_Z_ALERT_STRONG = 1.5
RMSSD_Z_SUPPRESS_ALERT = -1.0

# Cap RMSSD spike false alerts when change vs recent median exceeds this ratio.
RMSSD_SPIKE_RATIO_THRESHOLD = 0.40

# REST motion gate: acc_p90 below this (m/s^2) for spike filter.
REST_ACCELERATION_P90_MAX = 0.5

# Cap normalized arousal during physical motion (approx Moderate on 0-1 scale).
MOTION_AROUSAL_CAP = 0.66

# ML deployment gates (LOSO evaluation).
ML_MIN_ACCURACY = 0.80
ML_MIN_F1 = 0.72

# Baevsky raw SI population reference (PMC10305391).
BAEVSKY_RAW_NORMAL_LOW = 50.0
BAEVSKY_RAW_NORMAL_HIGH = 150.0


class KubiosZoneId(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    ELEVATED = "elevated"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass(frozen=True)
class KubiosZone:
    zone_id: KubiosZoneId
    label: str
    band_id: str
    arousal_mid: float
    arousal_scale_10: int


# Upper bounds for sqrt(Baevsky) stress_index -> zone mapping.
STRESS_INDEX_ZONE_BOUNDS = (
    (7.1, KubiosZoneId.LOW, "Relaxat", "low", 0.20, 2),
    (12.2, KubiosZoneId.NORMAL, "Echilibrat", "normal", 0.50, 5),
    (22.4, KubiosZoneId.ELEVATED, "Moderat", "elevated", 0.68, 7),
    (30.0, KubiosZoneId.HIGH, "Alert", "high", 0.82, 8),
    (float("inf"), KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 0.95, 10),
)


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
