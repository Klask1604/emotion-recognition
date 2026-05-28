"""
Signal-quality gate: deterministic, hysteresis-stabilised wrist HRV reliability.

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

Q is a deterministic function (no per-user online learning):
    Q = artifact_quality(A) * motion_penalty(motion_state)
The previous implementation also held an SGD logistic regression P(artifact|M)
that fed an anticipatory term. Removed: the gradient target was the *observed*
artifact rate (self-supervised noise), so the learned weights drifted without
a useful ground truth and the term mostly subtracted a small constant from Q
post-warm-up. The deterministic formula keeps the same shape (smooth degrade)
without unverified state.

Outputs:
  quality Q in [0, 1] — the decision-confidence component contributed by HRV.
  usable  = A <= A_MAX and motion_state == "still" (Kubios-aligned cutoff).
  motion_state = still / moving from the robust energy baseline + hysteresis.
"""

from __future__ import annotations

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
)


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


@dataclass
class SignalQualityState:
    """Per-subject mutable state for the still/moving baseline."""

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
    quality: float          # Q in [0, 1], the HRV-channel confidence
    usable: bool            # A <= ARTIFACT_RATE_MAX and not moving
    artifact_rate: float
    motion_energy: float
    p_artifact: float       # legacy slot, kept for payload compatibility (always 0)
    motion_state: str       # "still" | "moving"


def update_and_score(
    *,
    motion_energy: float,
    artifact_rate: float,
    state: SignalQualityState,
    has_signal: bool = True,
) -> SignalQuality:
    """Score the current epoch and update the motion baseline.

    `has_signal=False` means the IBI buffer was empty / below the HRV minimum:
    no beats were evaluated, so artifact_rate=0.0 is a degenerate value (0/0
    is not 'perfect'). We still classify motion (for the running baseline) and
    return Q=0 + usable=False so downstream UIs do NOT show fake-high
    confidence in nothing.
    """
    m = max(0.0, float(motion_energy))
    if not has_signal:
        # Keep the motion baseline current — that estimator is independent of
        # IBI data and otherwise stalls during silent periods.
        motion_state = state.classify_motion(m)
        state.motion_energy.append(m)
        return SignalQuality(
            quality=0.0,
            usable=False,
            artifact_rate=0.0,
            motion_energy=m,
            p_artifact=0.0,
            motion_state=motion_state,
        )
    a = min(1.0, max(0.0, float(artifact_rate)))

    # Classify motion BEFORE appending so the current sample does not mask
    # itself when it is an outlier (the same epoch's value would shift the
    # baseline used to evaluate it).
    motion_state = state.classify_motion(m)
    state.motion_energy.append(m)
    moving = motion_state == "moving"

    # Smooth confidence that degrades gradually (1 at zero artifacts, 0.5 at the
    # reliability cutoff) instead of snapping to 0 the moment A exceeds it — so
    # the displayed confidence is informative, not stuck at 0%.
    artifact_quality = 1.0 / (1.0 + (a / ARTIFACT_RATE_MAX) ** 2)
    quality = artifact_quality
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
        p_artifact=0.0,
        motion_state=motion_state,
    )
