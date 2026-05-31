"""
Wrist HRV module: the primary arousal channel from inter-beat intervals.

Requires the IBI capability. Produces a FusionChannel whose z is the personal
stress-index z-score and whose weight is the signal quality Q (HRV is precise
when the optical signal is clean/still). Pure wrapper over shared baseline +
fusion; the HRV math itself lives in shared/hrv.
"""

from __future__ import annotations

from affectus.contract.capabilities import Capability
from affectus.shared.fusion import FusionChannel

REQUIRES = frozenset({Capability.IBI})
NAME = "hrv"


def evaluate(ctx) -> FusionChannel | None:
    stress_z = (
        ctx.baseline.stress_index_z_score(ctx.primary.kubios_stress_index)
        if ctx.baseline.is_ready else 0.0
    )
    return FusionChannel(
        NAME, z=stress_z, weight=ctx.quality.quality, confidence=ctx.quality.quality
    )
