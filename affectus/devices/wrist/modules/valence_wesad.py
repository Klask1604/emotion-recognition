"""
Wrist valence module (WESAD-trained). Requires IBI + PPG.

Predicts positive/negative valence from the raw PPG window and contributes a
capped, confidence-weighted z. Default-disabled (VALENCE_WESAD_MAX_WEIGHT = 0)
and proven not to transfer to the watch (cross-device domain shift), so it is an
exact no-op in production — kept here only so the capability-gating is uniform.
The model + extractor live in engine/channels/valence_wesad (research code).
"""

from __future__ import annotations

from affectus.contract.capabilities import Capability
from affectus.shared.fusion import FusionChannel

REQUIRES = frozenset({Capability.IBI, Capability.PPG})
NAME = "valence_wesad"
# Disabled by default (cap 0, cross-device domain shift). Excluded from the
# handshake's announced module list so the watch is not told it is active.
ANNOUNCED = False


def evaluate(ctx) -> FusionChannel | None:
    if getattr(ctx, "valence_wesad", None) is None or not getattr(ctx, "ppg_green", None):
        return None
    from affectus.engine.channels.valence_wesad import (
        evaluate_valence_wesad,
        valence_weight,
    )

    res = evaluate_valence_wesad(ctx.valence_wesad, ctx.ppg_green, ctx.ppg_ts_ms or [])
    if res is None:
        return None
    weight = valence_weight(res.confidence)
    if weight <= 0.0:
        return None
    return FusionChannel(NAME, z=res.z, weight=weight, confidence=res.confidence)
