"""
All numeric thresholds, Kubios zone definitions, and on-disk paths for Biofizic.

Each block is annotated with its source so the thesis can defend every value:
  - LITERATURE: directly from a published reference (Kubios, Baevsky, etc.)
  - EMPIRICAL:  tuned on the Galaxy Watch 7 during development; documented in
                docs/THESIS_LIMITATIONS.md and not claimed to be universal
  - INFRA:      transport / publish-rate plumbing; not a physiological choice
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path


# ---------------------------------------------------------------------------
# IBI filtering and HRV preconditions (LITERATURE)
# ---------------------------------------------------------------------------

# Physiological IBI range. Beats outside [300, 2000] ms cannot come from a
# normal sinus rhythm and are rejected before HRV math runs.
MIN_INTERBEAT_INTERVAL_MS = 300
MAX_INTERBEAT_INTERVAL_MS = 2000

# Median-based artifact rejection. Any IBI that deviates from the running
# median by more than this ratio is dropped. Standard 20% rule used by
# Kubios HRV and most wrist-worn pipelines.
OUTLIER_MEDIAN_DEVIATION_RATIO = 0.20

# Reconstructed IBI timestamps must match the inter-beat delta within this
# tolerance, otherwise we fall back to consecutive-delta math. Picked to
# absorb Samsung Health SDK jitter without accepting badly anchored beats.
MAX_TIMESTAMP_IBI_MISMATCH_MS = 250

# Minimum data needed to publish a valid HRV epoch.
MIN_BEATS_FOR_HRV = 8
MIN_COVERED_SECONDS_FOR_HRV = 6.0

# Keep ~2 minutes of IBI so the 90 s validation window always has data
# regardless of small gaps.
IBI_BUFFER_RETENTION_MS = 120_000

# Histogram bin width for Baevsky stress index (AMo computation).
BAEVSKY_HISTOGRAM_BIN_MS = 50


# ---------------------------------------------------------------------------
# Analysis windows (LITERATURE for w30; w60/w90 EMPIRICAL validation)
# ---------------------------------------------------------------------------

# w30 drives every published decision. w60 and w90 are computed in parallel
# and surfaced only on biofizic/state/windows so the thesis can compare and
# justify why w30 is the right primary window for wrist HRV.
ANALYSIS_WINDOW_SECONDS = (30, 60, 90)
PRIMARY_DECISION_WINDOW_SECONDS = 30

# How often the full epoch payload is published on biofizic/state (retained).
EPOCH_PUBLISH_INTERVAL_SECONDS = 30

# How often the side-by-side window comparison is published. Lower than the
# epoch interval so Grafana shows fresh w60/w90 traces even mid-epoch.
WINDOWS_PUBLISH_INTERVAL_SECONDS = 5.0


# ---------------------------------------------------------------------------
# Live UI hysteresis (INFRA)
# ---------------------------------------------------------------------------

# The watch UI receives biofizic/state/live at 1 Hz. The rolling 30 s HRV
# buffer recomputes the Kubios stress index every second, so a single IBI
# entering or leaving the buffer can flip the integer arousal_10 across a
# zone boundary and the displayed number alternates 2-3-2-3 even when the
# signal is stable. The live stream only adopts a new integer after it has
# been seen for LIVE_AROUSAL_HYSTERESIS_TICKS consecutive ticks. Epoch
# decisions on biofizic/state stay reactive (no smoothing) so logs reflect
# the raw HRV at that tick.
LIVE_AROUSAL_HYSTERESIS_TICKS = 2


# ---------------------------------------------------------------------------
# Personal baseline (EMPIRICAL)
# ---------------------------------------------------------------------------

# Number of STILL epochs collected before the personal RMSSD/SI baseline is
# locked in. Four epochs at 30 s each gives ~2 minutes of resting data, which
# is the shortest interval where the median is reasonably stable on GW7.
STILL_EPOCHS_BEFORE_BASELINE_LOCK = 4

# EMA blend factor used after the baseline is locked. 0.05 lets slow drift
# follow circadian changes without being pulled by short bursts of activity.
BASELINE_EMA_ALPHA = 0.05


# ---------------------------------------------------------------------------
# Decision gate (EMPIRICAL, except where noted)
# ---------------------------------------------------------------------------

# Consecutive elevated epochs required before the gate confirms a real alert.
# Picks up sustained sympathetic activation while rejecting one-epoch noise.
ALERT_CONFIRMATION_EPOCH_COUNT = 2

# Z-score thresholds for the personal baseline. Z >= 1.0 signals deviation,
# Z >= 1.5 signals strong deviation and can confirm an alert on its own
# without needing the RMSSD veto. These are standard sigma thresholds; the
# personalisation (per-subject baseline) is the empirical part.
STRESS_INDEX_Z_ALERT = 1.0
STRESS_INDEX_Z_ALERT_STRONG = 1.5

# If RMSSD is at or below this baseline z-score the alert path is suppressed.
# Catches the case where SI rose only because the buffer just lost a few
# beats; without RMSSD also pointing the same direction we treat it as noise.
RMSSD_Z_SUPPRESS_ALERT = -1.0

# Single-tick RMSSD swings beyond this fraction of the recent median, while
# the subject is in REST, are capped. Protects against PPG artefacts where
# one beat is misread and RMSSD spikes.
RMSSD_SPIKE_RATIO_THRESHOLD = 0.40

# Cap on arousal_10 published while an alert is being confirmed. Display
# never crosses into the ALERT range until ALERT_CONFIRMATION_EPOCH_COUNT
# epochs have agreed. 6 keeps the watch on "Moderat" during the wait.
ALERT_PENDING_CAP = 6

# Considered REST when HAR says STILL and 90th-percentile wrist acceleration
# is below this value (m/s^2). The number is empirical, calibrated by
# observing GW7 acceleration at rest on the wrist; documented in
# docs/THESIS_LIMITATIONS.md.
REST_ACCELERATION_P90_MAX = 0.5


# ---------------------------------------------------------------------------
# Kubios zones (LITERATURE: Baevsky 1984, Kubios HRV User Guide)
# ---------------------------------------------------------------------------

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


# Stress index boundaries are the sqrt-Baevsky breakpoints used by Kubios.
# Romanian display labels stay as-is because they are the UI strings the
# watch face shows; do not translate them without updating the watch.
STRESS_INDEX_ZONE_BOUNDS = (
    (7.1, KubiosZoneId.LOW, "Relaxat", "low", 0.20, 2),
    (12.2, KubiosZoneId.NORMAL, "Echilibrat", "normal", 0.50, 5),
    (22.4, KubiosZoneId.ELEVATED, "Moderat", "elevated", 0.68, 7),
    (30.0, KubiosZoneId.HIGH, "Alert", "high", 0.82, 8),
    (float("inf"), KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 0.95, 10),
)


def clip_value(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


# ---------------------------------------------------------------------------
# Motion / HAR (EMPIRICAL caps applied to a literature-trained classifier)
# ---------------------------------------------------------------------------

# Per-class caps on the displayed arousal_10. The WISDM HAR classifier is
# itself trained on a public dataset, but the caps that say "if the user is
# walking we cannot trust HRV above 5/10" are an empirical guard against
# motion-induced PPG artefacts.
HAR_AROUSAL_CAP_BY_CLASS = {
    "STILL": 10,
    "SCROLL": 6,
    "HAND": 6,
    "WALK": 5,
}

HAR_CLASS_NAMES = ("STILL", "SCROLL", "HAND", "WALK")


# ---------------------------------------------------------------------------
# Paths (INFRA)
# ---------------------------------------------------------------------------

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
