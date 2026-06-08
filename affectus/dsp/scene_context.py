"""Map a VR scene's visual context to a valence prior.

Valence cannot be read from wrist PPG (proven). In VR we instead DERIVE it from
what the scene looks like: the Unity SceneContextProbe measures raw visual cues
(brightness, dominant colour, motion) and the server maps them here to a valence
prior in [-1, +1]. This is an HONEST multimodal approach, the verdict combines
the MEASURED arousal (from physiology) with the SCENE's valence prior (context),
and the published emotion is marked valence_source="context" so it never pretends
to read valence from the pulse.

The mapping is deliberately transparent and rule-based (no hidden model), so it
can be inspected and tuned. Grounded in well-known affective-design heuristics:
  - Bright, green/blue scenes (nature, daylight) read as pleasant  -> +.
  - Dark, red-dominant scenes (danger, blood, alarm) read as unpleasant -> -.
  - High visual motion/chaos pushes toward the negative/threat side at equal light.
Everything is a PRIOR, not a measurement, it biases the valence axis of the
Russell quadrant, and physiology still drives arousal.
"""

from __future__ import annotations


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def context_to_valence_prior(
    light: float,
    r: float,
    g: float,
    b: float,
    motion: float,
) -> float:
    """Raw scene cues -> valence prior in [-1, +1].

    light  : ambient brightness, 0 (dark) .. 1 (bright).
    r,g,b  : dominant scene colour, each 0..1 (skybox/fog/ambient tint).
    motion : visual motion intensity, 0 (static) .. 1 (chaotic).

    Returns >0 for pleasant-looking scenes, <0 for threatening ones, 0 if neutral
    or inputs are missing/degenerate.
    """
    # Guard: degenerate / missing -> neutral prior (fail-safe to 0).
    if not all(isinstance(x, (int, float)) for x in (light, r, g, b, motion)):
        return 0.0

    # 1. Brightness: bright = pleasant, dark = unpleasant. Centred at 0.5.
    light_term = (float(light) - 0.5) * 1.2          # ~[-0.6, +0.6]

    # 2. Colour tone: green/blue calm-pleasant, red alarm-unpleasant.
    #    cool = (g + b) favours +, warm = r favours -.
    cool = (float(g) + float(b)) / 2.0
    warm = float(r)
    colour_term = (cool - warm) * 0.8                # ~[-0.8, +0.8]

    # 3. Motion: high visual chaos biases toward threat/negative.
    motion_term = -float(motion) * 0.5               # [-0.5, 0]

    prior = light_term + colour_term + motion_term
    return round(_clamp(prior), 3)


def valence_prior_label(prior: float, deadband: float = 0.15) -> str:
    """Human-readable tag for the prior (for logs/dashboard, not the verdict)."""
    if prior > deadband:
        return "pleasant"
    if prior < -deadband:
        return "unpleasant"
    return "neutral"
