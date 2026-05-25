"""Map RMSSD z-score and PPG pulse amplitude z into a valence scale."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ValenceResult:
    """Affect valence derived from autonomic proxies."""

    valence_10: int
    valence_label: str
    affect_quadrant: str
    valence_score: float


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def valence_label_from_score(valence_10: int) -> str:
    """Map 1..10 valence scale to a short label."""
    if valence_10 <= 3:
        return "negative"
    if valence_10 <= 6:
        return "neutral"
    return "positive"


def affect_quadrant_from_axes(arousal_10: int, valence_10: int) -> str:
    """
    Russell circumplex quadrant from arousal (1..10) and valence (1..10).
    Midpoint at 5 for both axes.
    """
    high_arousal = arousal_10 >= 6
    high_valence = valence_10 >= 6
    if high_arousal and high_valence:
        return "activated"
    if high_arousal and not high_valence:
        return "tense"
    if not high_arousal and high_valence:
        return "calm"
    return "depleted"


def compute_valence(
    *,
    rmssd_z_score: float,
    z_pulse_amp: float,
    motion_class: str,
    baseline_ready: bool,
) -> ValenceResult:
    """
    Heuristic valence from parasympathetic (RMSSD z) and sympathetic (PPG amp z).

    rmssd_z_score: positive when RMSSD is above personal baseline (parasympathetic).
    z_pulse_amp: positive when pulse amplitude drops vs rest (sympathetic vasoconstriction).
    motion_class: valence is damped when not STILL (noisy PPG / context).
    """
    if not baseline_ready:
        return ValenceResult(
            valence_10=5,
            valence_label="neutral",
            affect_quadrant="pending",
            valence_score=0.0,
        )

    rmssd_term = math.tanh(rmssd_z_score / 2.0)
    pulse_term = math.tanh(z_pulse_amp / 2.0)
    valence_score = _clip(0.55 * rmssd_term - 0.35 * pulse_term, -1.0, 1.0)

    if motion_class != "STILL":
        valence_score *= 0.75

    valence_10 = int(round(_clip(5.0 + valence_score * 4.0, 1.0, 10.0)))
    label = valence_label_from_score(valence_10)
    return ValenceResult(
        valence_10=valence_10,
        valence_label=label,
        affect_quadrant="pending",
        valence_score=valence_score,
    )


def finalize_affect_quadrant(valence: ValenceResult, arousal_10: int) -> ValenceResult:
    """Attach quadrant after display arousal is known."""
    return ValenceResult(
        valence_10=valence.valence_10,
        valence_label=valence.valence_label,
        affect_quadrant=affect_quadrant_from_axes(arousal_10, valence.valence_10),
        valence_score=valence.valence_score,
    )
