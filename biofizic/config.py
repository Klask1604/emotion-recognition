"""Constants, paths, and legacy threshold re-exports for Biofizic."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# --- HRV ---

MIN_INTERBEAT_INTERVAL_MS = 300
MAX_INTERBEAT_INTERVAL_MS = 2000
OUTLIER_MEDIAN_DEVIATION_RATIO = 0.20
MAX_TIMESTAMP_IBI_MISMATCH_MS = 250
MIN_BEATS_FOR_HRV = 8
MIN_COVERED_SECONDS_FOR_HRV = 6.0
IBI_LOOKBACK_TRIM_MS = 60_000
IBI_BUFFER_RETENTION_MS = 120_000
BAEVSKY_HISTOGRAM_BIN_MS = 50
SQRT_BAEVSKY_NORMAL_LOW = 7.0
SQRT_BAEVSKY_NORMAL_HIGH = 12.0
ANALYSIS_WINDOW_SECONDS = (15, 30, 60, 90)
PRIMARY_DECISION_WINDOW_SECONDS = 30
EPOCH_PUBLISH_INTERVAL_SECONDS = 30

# --- Kubios zones ---

STILL_EPOCHS_BEFORE_BASELINE_LOCK = 4
BASELINE_EMA_ALPHA = 0.05
ALERT_CONFIRMATION_EPOCH_COUNT = 2
STRESS_INDEX_Z_ALERT = 1.0
STRESS_INDEX_Z_ALERT_STRONG = 1.5
RMSSD_Z_SUPPRESS_ALERT = -1.0
RMSSD_SPIKE_RATIO_THRESHOLD = 0.40
REST_ACCELERATION_P90_MAX = 0.5
MOTION_AROUSAL_CAP = 0.66
ML_MIN_ACCURACY = 0.80
ML_MIN_F1 = 0.72
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


STRESS_INDEX_ZONE_BOUNDS = (
    (7.1, KubiosZoneId.LOW, "Relaxat", "low", 0.20, 2),
    (12.2, KubiosZoneId.NORMAL, "Echilibrat", "normal", 0.50, 5),
    (22.4, KubiosZoneId.ELEVATED, "Moderat", "elevated", 0.68, 7),
    (30.0, KubiosZoneId.HIGH, "Alert", "high", 0.82, 8),
    (float("inf"), KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 0.95, 10),
)


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# --- Motion ---

HAR_AROUSAL_CAP_BY_CLASS = {
    "STILL": 10,
    "SCROLL": 6,
    "HAND": 6,
    "WALK": 5,
}

HAR_CLASS_NAMES = ("STILL", "SCROLL", "HAND", "WALK")
ACTIVITY_CONTEXT_WINDOW_SAMPLES = 10

# --- Paths ---

ROOT = Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    docker = Path("/data")
    if docker.is_dir():
        return docker
    local = ROOT / "data"
    local.mkdir(parents=True, exist_ok=True)
    return local


def models_dir() -> Path:
    docker_models = Path("/app/models")
    if docker_models.is_dir():
        return docker_models
    local = ROOT / "models"
    local.mkdir(parents=True, exist_ok=True)
    return local


def user_profile_path() -> Path:
    return data_dir() / "user_profile.json"


def user_baseline_path() -> Path:
    return data_dir() / "user_baseline.json"


def rest_baseline_path() -> Path:
    return data_dir() / "rest_baseline.json"


def motion_model_path() -> Path:
    return data_dir() / "motion_model.json"


def model_v3_path() -> Path:
    v4 = models_dir() / "model_v4.joblib"
    if v4.exists():
        return v4
    return models_dir() / "model_v3.joblib"


def model_v4_path() -> Path:
    return models_dir() / "model_v4.joblib"


def population_stats_path() -> Path:
    return models_dir() / "population_stats.json"


# --- Legacy thresholds (backward-compatible aliases) ---

from biofizic.decision.arousal_mapper import (
    KubiosZone as ZoneInfo,
    arousal_scale_10_to_label as arousal_10_to_label,
    baseline_z_score_to_label as z_si_to_label,
    kubios_zone_for_stress_index as kubios_zone,
    stress_index_to_arousal,
    zone_is_alert_or_higher,
    zone_is_elevated_or_higher,
)

ALERT_CONFIRM_EPOCHS = ALERT_CONFIRMATION_EPOCH_COUNT
BAEVSKY_NORMAL_RAW_HI = BAEVSKY_RAW_NORMAL_HIGH
BAEVSKY_NORMAL_RAW_LO = BAEVSKY_RAW_NORMAL_LOW
PASSIVE_EMA_ALPHA = BASELINE_EMA_ALPHA
ML_GATE_ACCURACY_MIN = ML_MIN_ACCURACY
ML_GATE_F1_MIN = ML_MIN_F1
REST_ACC_P90_MAX = REST_ACCELERATION_P90_MAX
RMSSD_SPIKE_RATIO = RMSSD_SPIKE_RATIO_THRESHOLD
Z_RMSSD_SUPPRESS = RMSSD_Z_SUPPRESS_ALERT
PASSIVE_REST_EPOCHS_MIN = STILL_EPOCHS_BEFORE_BASELINE_LOCK
Z_SI_ALERT = STRESS_INDEX_Z_ALERT
Z_SI_ALERT_STRONG = STRESS_INDEX_Z_ALERT_STRONG
KubiosSiZone = KubiosZoneId

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
