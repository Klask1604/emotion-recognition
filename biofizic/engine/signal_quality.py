"""
Signal-quality gate: a physically grounded replacement for HAR arousal caps.

The validity of wrist HRV is limited by motion contaminating the optical PPG
and by the IBI artifacts that result. We measure both directly instead of
inferring an activity class:

  M = acceleration power in the cardiac band (0.5-4 Hz), reported per batch by
      the watch. This is the band that overlaps the PPG pulse, i.e. the part of
      wrist motion that actually corrupts the optical signal (cf. accelerometer
      motion-artifact removal, TROIKA, Zhang et al. 2015).
  A = IBI artifact rate, the fraction of beats rejected by the physiological /
      outlier filter over the analysis window. A standard HRV signal-quality
      index (Task Force ESC/NASPE 1996).

still vs moving is decided from M against the subject's own resting motion
baseline (robust median + MAD), NOT from A: at rest the wrist is nearly
motionless (M ~ 0) yet wrist PPG still yields IBI artifacts, so the artifact
rate is not a motion proxy.

The motion -> artifact relationship is additionally learned per subject by
online logistic regression P(artifact | M) = sigma(b0 + b1*M) and feeds the
anticipatory term of the quality score.

Outputs:
  quality Q in [0, 1] = (1 - min(A / A_MAX, 1)) * (1 - P(artifact | M))
      published as the decision confidence.
  usable  = A <= A_MAX (Kubios-aligned reliability cutoff).
  motion_state = still / moving.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

from biofizic.config import (
    ARTIFACT_RATE_MAX,
    MIN_QUALITY_UPDATES,
    MOTION_BASELINE_WINDOW,
    MOTION_ENERGY_SIGMA_FLOOR,
    MOTION_ENERGY_SIGMA_REL,
    MOTION_ENTER_SIGMA,
    MOTION_EXIT_SIGMA,
    MOTION_MOVING_QUALITY_FACTOR,
    QUALITY_LOGISTIC_LEARNING_RATE,
)


def _sigmoid(x: float) -> float:
    # Numerically stable logistic.
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _logit(p: float) -> float:
    p = min(max(p, 1e-6), 1.0 - 1e-6)
    return math.log(p / (1.0 - p))


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


@dataclass
class SignalQualityState:
    """Per-subject mutable state across epochs."""

    # Logistic weights for the anticipatory P(artifact | motion) term.
    b0: float = _logit(ARTIFACT_RATE_MAX)
    b1: float = 0.0
    n_updates: int = 0
    # Rolling window of recent motion energy for the robust still/moving baseline.
    motion_energy: deque[float] = field(
        default_factory=lambda: deque(maxlen=MOTION_BASELINE_WINDOW)
    )
    motion_is_moving: bool = False  # hysteresis state

    def classify_motion(self, energy: float) -> str:
        """still / moving from the motion energy's own robust baseline, with
        hysteresis so micro-noise cannot flip the label."""
        vals = list(self.motion_energy)
        if len(vals) < MIN_QUALITY_UPDATES:
            self.motion_is_moving = False
            return "still"  # cold start: assume still until we have a baseline
        center = _median(vals)
        mad = _median([abs(v - center) for v in vals])
        sigma = max(1.4826 * mad, MOTION_ENERGY_SIGMA_REL * center, MOTION_ENERGY_SIGMA_FLOOR)
        if self.motion_is_moving:
            # Stay moving until clearly back down (lower band).
            if energy < center + MOTION_EXIT_SIGMA * sigma:
                self.motion_is_moving = False
        else:
            # Become moving only on a clear, sustained excursion (upper band).
            if energy > center + MOTION_ENTER_SIGMA * sigma:
                self.motion_is_moving = True
        return "moving" if self.motion_is_moving else "still"


@dataclass(frozen=True)
class SignalQuality:
    quality: float          # Q in [0, 1], used as decision confidence
    usable: bool            # A <= ARTIFACT_RATE_MAX
    artifact_rate: float
    motion_energy: float
    p_artifact: float       # learned P(artifact | M) after the update
    motion_state: str       # "still" | "moving"


def update_and_score(
    *,
    motion_energy: float,
    artifact_rate: float,
    state: SignalQualityState,
) -> SignalQuality:
    """Update the per-user model with this epoch and return its quality."""
    m = max(0.0, float(motion_energy))
    a = min(1.0, max(0.0, float(artifact_rate)))

    # One SGD step of logistic regression using the observed artifact rate as a
    # soft target. d/db (cross-entropy) = (p - a) * [1, m].
    p = _sigmoid(state.b0 + state.b1 * m)
    grad = p - a
    lr = QUALITY_LOGISTIC_LEARNING_RATE
    state.b0 -= lr * grad
    state.b1 -= lr * grad * m
    state.n_updates += 1
    p_after = _sigmoid(state.b0 + state.b1 * m)

    # Motion state from the energy's own baseline (independent of artifacts),
    # classified before appending so the current sample does not mask itself.
    motion_state = state.classify_motion(m)
    state.motion_energy.append(m)
    moving = motion_state == "moving"

    # Smooth confidence that degrades gradually (1 at zero artifacts, 0.5 at the
    # reliability cutoff) instead of snapping to 0 the moment A exceeds it — so
    # the displayed confidence is informative, not stuck at 0%.
    artifact_quality = 1.0 / (1.0 + (a / ARTIFACT_RATE_MAX) ** 2)
    quality = artifact_quality * (1.0 - p_after)
    if moving:
        # Motion corrupts wrist PPG even when beats stay in range (low artifact
        # rate but wrong RMSSD), so a moving epoch is not trustworthy.
        quality *= MOTION_MOVING_QUALITY_FACTOR
    usable = (a <= ARTIFACT_RATE_MAX) and not moving

    return SignalQuality(
        quality=quality,
        usable=usable,
        artifact_rate=a,
        motion_energy=m,
        p_artifact=p_after,
        motion_state=motion_state,
    )
