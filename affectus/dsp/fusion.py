"""
Fusion core: the device-agnostic math that combines arousal channels into one
smoothed, change-detected verdict. Shared by every device family — a wrist, a
chest strap or a head-worn device all feed FusionChannels into _fuse and the same
Kalman/CUSUM. Only WHICH channels exist differs per device (devices/<family>),
not how they are fused.

Contents (moved verbatim out of engine/decision.py):
  - DecisionState      per-session Kalman + CUSUM state
  - _kalman_update     scalar Kalman, measurement variance = BASE / quality
  - _cusum_update      one-sided CUSUM (Page 1954), latched alert
  - FusionChannel      one arousal channel (name, z, weight, confidence)
  - _fuse              weighted mean of channel z and confidence
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.config import (
    CUSUM_SLACK_K,
    CUSUM_THRESHOLD_H,
    KALMAN_MEAS_VAR_BASE,
    KALMAN_PROCESS_VAR,
    KALMAN_QUALITY_FLOOR,
)


# ── State ────────────────────────────────────────────────────────────────────

@dataclass
class DecisionState:
    """Per-session mutable state owned by the decision module: Kalman estimate
    + variance, plus CUSUM accumulator + latched alert. Single dataclass so
    `pipeline.PhysiologyPipeline` carries one state object instead of three."""

    # Scalar Kalman on the personal stress z. x=0 means "at personal baseline".
    estimator_x: float = 0.0
    estimator_P: float = 1.0
    estimator_process_var: float = KALMAN_PROCESS_VAR

    # One-sided CUSUM on the filtered z. Latches True until the accumulator
    # decays back to zero, so the alert has built-in hysteresis (Page 1954).
    cusum_slack_k: float = CUSUM_SLACK_K
    cusum_threshold_h: float = CUSUM_THRESHOLD_H
    cusum_s: float = 0.0
    cusum_alert: bool = False

    def reset(self) -> None:
        self.estimator_x = 0.0
        self.estimator_P = 1.0
        self.cusum_s = 0.0
        self.cusum_alert = False


# ── Kalman + CUSUM ───────────────────────────────────────────────────────────

def _kalman_update(state: DecisionState, z_measured: float, quality: float) -> tuple[float, float]:
    """Fold one epoch's z into the Kalman estimate. Returns (x_filtered, gain).

    Measurement variance scales as BASE / max(Q, FLOOR): a low-quality epoch
    yields a large variance, tiny Kalman gain, so the estimate barely moves.
    This unifies the signal-quality gate and the older "hold last value" patch
    into one principled filter.
    """
    q = max(float(quality), KALMAN_QUALITY_FLOOR)
    r = KALMAN_MEAS_VAR_BASE / q
    state.estimator_P += state.estimator_process_var
    gain = state.estimator_P / (state.estimator_P + r)
    state.estimator_x += gain * (float(z_measured) - state.estimator_x)
    state.estimator_P *= 1.0 - gain
    return state.estimator_x, gain


def _cusum_update(state: DecisionState, z: float) -> bool:
    """One-sided CUSUM:   S_t = max(0, S_{t-1} + (z_t - k))
    Alerts when S_t > h; the alert latches until S_t decays back to 0."""
    state.cusum_s = max(0.0, state.cusum_s + (float(z) - state.cusum_slack_k))
    if state.cusum_s > state.cusum_threshold_h:
        state.cusum_alert = True
    elif state.cusum_s == 0.0:
        state.cusum_alert = False
    return state.cusum_alert


# ── Multi-channel fusion ─────────────────────────────────────────────────────

@dataclass(frozen=True)
class FusionChannel:
    """One arousal channel feeding the weighted-mean fusion.

    name:       diagnostic label ("hrv", "hr", "temp", "resp").
    z:          the channel's personal z-score (>0 = more aroused).
    weight:     how much this channel contributes to the fused z. For HRV this
                is the signal quality Q; for HR it is (1 - Q). New channels
                (temp, resp) enter with weight 0 until their own quality gate is
                wired in, which makes them an exact no-op against the current
                pipeline (see _fuse).
    confidence: the channel's own confidence, blended into fused_confidence with
                the same weights.
    """

    name: str
    z: float
    weight: float
    confidence: float


def _fuse(channels: list[FusionChannel]) -> tuple[float, float]:
    """Weighted mean of the channel z-scores and confidences.

        z_fused    = Σ wᵢ·zᵢ    / Σ wᵢ
        confidence = Σ wᵢ·confᵢ / Σ wᵢ

    With only HRV (weight Q) and HR (weight 1-Q) the denominator is exactly 1,
    so this reduces bit-for-bit to the previous
        z_fused = Q·z_hrv + (1-Q)·z_hr
    The normalisation only changes anything once a third channel adds weight,
    at which point every channel is re-weighted to keep z_fused on the same
    z-score scale. Returns (0, 0) if no channel carries weight."""
    total_w = sum(c.weight for c in channels)
    if total_w <= 0.0:
        return 0.0, 0.0
    z = sum(c.weight * c.z for c in channels) / total_w
    conf = sum(c.weight * c.confidence for c in channels) / total_w
    return z, conf
