"""
Wrist HR module: the motion-robust arousal channel from the SDK heart rate.

Requires the HR capability. Weight is (1 - Q): HR carries the verdict when the
optical signal is corrupted by motion and HRV quality Q drops. Confidence is a
fixed constant when HR is meaningful (baseline ready and a real reading present),
else 0 so the channel is an exact no-op.
"""

from __future__ import annotations

from affectus.config import HR_CHANNEL_CONFIDENCE
from affectus.contract.capabilities import Capability
from affectus.shared.fusion import FusionChannel

REQUIRES = frozenset({Capability.HR})
NAME = "hr"


def evaluate(ctx) -> FusionChannel | None:
    sdk_hr = ctx.sensor.heart_rate_bpm if ctx.sensor else 0.0
    hr_z = ctx.baseline.hr_z_score(sdk_hr) if ctx.baseline.is_ready else 0.0
    hr_present = sdk_hr > 0.0 and ctx.baseline.is_ready
    hr_conf = HR_CHANNEL_CONFIDENCE if hr_present else 0.0
    return FusionChannel(
        NAME, z=hr_z, weight=1.0 - ctx.quality.quality, confidence=hr_conf
    )
