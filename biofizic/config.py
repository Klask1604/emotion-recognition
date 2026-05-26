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
# beat is flagged as an artifact when it deviates from the LOCAL median (of its
# neighbours, see LOCAL_MEDIAN_HALF_WINDOW) by more than this ratio. Comparing
# to the LOCAL median, not the whole-window median, is what keeps genuine HRV
# (a wide-but-normal RR distribution) from being mislabelled as artifacts
# (Kubios HRV artifact correction). 20% is the Malik/Kubios threshold.
OUTLIER_MEDIAN_DEVIATION_RATIO = 0.20
# Half-width (in beats) of the neighbourhood used for the local median above.
LOCAL_MEDIAN_HALF_WINDOW = 5

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
# Raw PPG / peak detection (RESEARCH/LEGACY — see biofizic/legacy)
# ---------------------------------------------------------------------------

# Band-pass for the PPG pulse wave: the cardiac band. Below 0.5 Hz is baseline
# wander, above ~4 Hz is noise/motion. find_peaks then locates systolic peaks.
PPG_BAND_LO_HZ = 0.5
PPG_BAND_HI_HZ = 4.0
# Minimum samples in the analysis window before peak detection is attempted.
PPG_MIN_SAMPLES = 64
# Minimum spacing between beats (s): 300 ms == 200 bpm ceiling, matches the IBI
# physiological floor.
PPG_MIN_BEAT_DISTANCE_S = 0.3
# Rolling window the legacy raw-PPG engine analyses (s) and its PPA baseline.
PPG_ANALYSIS_WINDOW_S = 8.0
PPG_PPA_BASELINE_WINDOW = 60


# ---------------------------------------------------------------------------
# Analysis windows (LITERATURE for w30; w60/w90 EMPIRICAL validation)
# ---------------------------------------------------------------------------

# w30 drives every published decision. w60 and w90 are computed in parallel
# and surfaced only on biofizic/state/windows so the thesis can compare and
# justify why w30 is the right primary window for wrist HRV.
ANALYSIS_WINDOW_SECONDS = (30, 60, 90)
PRIMARY_DECISION_WINDOW_SECONDS = 30

# Lookback used when slicing the IBI buffer for the multi-window HRV pass:
# the longest analysis window, so every window has the data it needs.
HRV_LOOKBACK_MS = max(ANALYSIS_WINDOW_SECONDS) * 1000

# Minimum beats before any HRV statistic is computed at all (RMSSD needs >= 2
# successive intervals). Distinct from MIN_BEATS_FOR_HRV, which gates the
# stricter "full quality" verdict.
MIN_BEATS_FOR_ANY_HRV = 2

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
#
# Value chosen as 3 after observing that 2 still let pure 2-tick transitions
# through. With 3 a new integer must persist three live ticks (~3 s of wall
# clock) before the watch UI adopts it, which kills the 2-3-2-3 alternation
# while keeping real transitions visible within 3 s.
LIVE_AROUSAL_HYSTERESIS_TICKS = 3


# ---------------------------------------------------------------------------
# Signal quality / motion-artifact gate (LITERATURE + INFRA)
# ---------------------------------------------------------------------------

# Maximum IBI artifact rate (fraction of beats corrected over the window) for an
# epoch's HRV to be considered reliable enough to (a) update the baseline and
# (b) carry high confidence. NOTE: 5% is an ECG / chest-strap figure; consumer
# wrist PPG routinely runs 10-20% corrected beats even on a good recording, so a
# 5% gate is never met on the wrist -> confidence pinned near 0 AND the baseline
# never updates. 15% is a realistic wrist-PPG threshold (beats are still
# corrected by interpolation regardless; this only sets the trust cutoff).
ARTIFACT_RATE_MAX = 0.15

# Online logistic regression P(artifact | motion_energy) = sigma(b0 + b1*M),
# used only for the anticipatory term of the quality score Q. The motion->
# artifact relationship is learned per subject. INFRA (optimiser plumbing).
QUALITY_LOGISTIC_LEARNING_RATE = 0.05
# Samples collected before the still/moving classifier and the quality model
# are trusted; until then we assume "still" (cold start).
MIN_QUALITY_UPDATES = 8

# Still/moving is decided from the cardiac-band motion energy itself, NOT from
# the IBI artifact rate: at rest the wrist barely moves (energy ~ 0) yet wrist
# PPG still produces IBI artifacts, so artifacts are not a motion proxy. We keep
# a robust baseline (median + MAD) of the motion energy over a rolling window
# and flag "moving" only when the current energy is a clear upper outlier.
MOTION_BASELINE_WINDOW = 120          # ~2 min at 1 Hz
# Hysteresis so the still/moving label does not flip on micro-noise: enter
# "moving" only on a clear excursion (ENTER sigma) and return to "still" once
# it falls back below the lower (EXIT) band.
MOTION_ENTER_SIGMA = 4.0
MOTION_EXIT_SIGMA = 2.0
# Robust sigma floor for the motion-energy baseline: max of a small absolute
# value and a fraction of the resting median (scale-free across acc_rms vs
# acc_band_cardiac). MAD collapses to 0 when the wrist is consistently still.
MOTION_ENERGY_SIGMA_FLOOR = 0.02
MOTION_ENERGY_SIGMA_REL = 0.5
# When the wrist is moving the epoch is marked unusable AND its quality is
# multiplied by this factor, so a motion-corrupted RMSSD (low artifact rate but
# wrong) barely moves the Kalman estimate. Motion contaminates wrist PPG even
# when individual beats stay in physiological range.
MOTION_MOVING_QUALITY_FACTOR = 0.1


# ---------------------------------------------------------------------------
# Personal baseline (EMPIRICAL)
# ---------------------------------------------------------------------------

# Robust personal baseline (log-space). RMSSD and the Kubios SI are log-normal
# (Task Force ESC/NASPE 1996; Nunan 2010), so the baseline is estimated on
# ln(x) with a robust location/scale: median and MAD, with sigma = 1.4826*MAD
# (Hampel). z = (ln x - median) / sigma. This replaces the earlier fixed 15%
# scale, which was not a statistic.
#
# Lock after this many resting epochs (~6 min at 30 s); fewer is too noisy for
# a stable MAD. The estimate then slides over a rolling window of resting
# epochs, which adapts to circadian drift without an EMA fudge factor.
BASELINE_MIN_REST_EPOCHS = 12
BASELINE_ROBUST_WINDOW_EPOCHS = 60
# Numerical floor on the log-space sigma (~5% multiplicative) so z is finite
# when a subject's resting HRV is unusually stable. INFRA, not physiological.
BASELINE_LOG_SIGMA_FLOOR = 0.05
# Clip the reported z-score; |z| > 4 is past the usable range of the CDF map.
Z_SCORE_CLIP = 4.0

# Scalar Kalman smoother on the personal stress z (engine/state_estimator.py).
# The latent autonomic state drifts slowly between 30 s epochs (small process
# variance); each epoch is a noisy measurement whose variance scales inversely
# with the signal quality Q, so artifact/motion-heavy epochs barely move the
# estimate. This is the state-space replacement for the ad-hoc "hold last value"
# patch (Kalman 1960). The process/measurement variance ratio sets how fast the
# estimate follows real changes — these are the tunable smoothing knobs, not
# decision thresholds.
KALMAN_PROCESS_VAR = 0.02       # latent z drift per epoch
KALMAN_MEAS_VAR_BASE = 0.5      # measurement variance at perfect quality (Q=1)
KALMAN_QUALITY_FLOOR = 0.05     # floor on Q so meas variance stays finite


# ---------------------------------------------------------------------------
# Motion-tolerant fusion (VR context classifier)
# ---------------------------------------------------------------------------

# The arousal measurement fed to the Kalman is a quality-weighted blend of the
# HRV-based personal z (precise when still) and an HR-based personal z (robust
# to wrist motion). z_fused = w*z_si + (1-w)*z_hr, with w = signal quality Q.
# In a VR club/boxing scene (lots of motion) the HRV part is unreliable but HR
# stays informative, so the system reports genuine high arousal (exertion /
# excitement) instead of freezing. Motion is a context feature, never a veto.
#
# Confidence of the HR channel: HR is SDK-processed and fairly robust even in
# motion, so the fused measurement keeps a usable confidence floor (the Kalman
# still updates during motion, just leaning on HR).
HR_CHANNEL_CONFIDENCE = 0.7

# Which channel is reported as dominant for the verdict, from the HRV weight
# (= signal quality). Above HIGH the verdict is HRV-driven (still, precise);
# below LOW it is HR-driven (motion, robust); in between it is a blend. Used for
# display/diagnostics only, not for the math (the fusion is continuous).
CHANNEL_HRV_DOMINANT_ABOVE = 0.6
CHANNEL_HR_DOMINANT_BELOW = 0.4

# Watch->server skew above this (seconds) is treated as a delivery backlog and
# warned about. A steady ~1-2 s is normal pipeline latency (batch buffering on
# the watch + MQTT), so we do not warn below it.
SKEW_BACKLOG_WARN_SEC = 5.0


# ---------------------------------------------------------------------------
# Decision gate: CUSUM alert detector (LITERATURE)
# ---------------------------------------------------------------------------

# Sustained stress is confirmed by a one-sided CUSUM on the personal
# stress-index z-score (Page 1954; Montgomery SPC), replacing the old
# "N consecutive elevated epochs" counter and the RMSSD spike / dual-veto
# heuristics (artifacts are now handled by the signal-quality gate). Because z
# is already in sigma units, the textbook SPC defaults apply directly:
#   k = reference value (slack) the signal must persistently exceed,
#   h = decision interval at which the accumulated evidence triggers an alert.
CUSUM_SLACK_K = 0.5
CUSUM_THRESHOLD_H = 4.0


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
    arousal_scale_10: int  # representative population-zone severity (1..10)


# Stress index boundaries are the sqrt-Baevsky breakpoints used by Kubios.
# arousal_scale_10 is the zone's severity level used only as the pre-baseline
# fallback (before personal calibration); the live arousal is the normal-CDF
# of the personal z-score (see arousal_mapper). Romanian display labels stay
# as-is because they are the UI strings the watch face shows; do not translate
# them without updating the watch.
STRESS_INDEX_ZONE_BOUNDS = (
    (7.1, KubiosZoneId.LOW, "Relaxat", "low", 2),
    (12.2, KubiosZoneId.NORMAL, "Echilibrat", "normal", 5),
    (22.4, KubiosZoneId.ELEVATED, "Moderat", "elevated", 7),
    (30.0, KubiosZoneId.HIGH, "Alert", "high", 8),
    (float("inf"), KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 10),
)


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


def rest_baseline_path() -> Path:
    return data_dir() / "rest_baseline.json"
