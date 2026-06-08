"""
Fusion channels, builds the per-epoch arousal channels for a wrist frame.

This replaces the old device-module registry (devices/registry.py + one file per
module under devices/wrist/modules/). There is one real device (the wrist watch),
so a plug-in registry that fanned out over wrist/chest/head module lists to run
three functions was machinery without a payoff. The capability CONTRACT stays
(contract/capabilities.py), chest/head devices will declare what they carry, but
assembling the channels is now a flat, readable function: HRV and HR always, plus
temperature when the frame carries a skin-temperature reading.

Two gates, same as before, kept explicit:
  - signal present?  -> build the channel at all (e.g. temp only if SKIN_TEMP)
  - weight > 0?      -> the channel actually contributes (the math returns None)
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.config import HR_CHANNEL_CONFIDENCE, TEMP_CHANNEL_MAX_WEIGHT
from affectus.contract.capabilities import Capability
from affectus.dsp.fusion import FusionChannel
from affectus.dsp.temperature import evaluate_skin_temperature


@dataclass
class ChannelContext:
    """Everything a channel builder may need for one epoch. Assembled by the
    pipeline; each builder reads only what its signal grants."""

    primary: object                      # HrvMetrics
    sensor: object | None                # AcquisitionBatchMessage (latest)
    quality: object                      # SignalQuality
    baseline: object                     # RestBaselineStore
    temperature: object | None = None    # SkinTemperatureChannelState
    present: frozenset[Capability] = frozenset()


def hrv_channel(ctx: ChannelContext) -> FusionChannel:
    """Primary arousal channel from inter-beat intervals. z is the personal
    stress-index z-score; weight is the signal quality Q (HRV is precise when the
    optical signal is clean/still)."""
    stress_z = (
        ctx.baseline.stress_index_z_score(ctx.primary.kubios_stress_index)
        if ctx.baseline.is_ready else 0.0
    )
    return FusionChannel(
        "hrv", z=stress_z, weight=ctx.quality.quality, confidence=ctx.quality.quality
    )


def hr_channel(ctx: ChannelContext) -> FusionChannel:
    """Motion-robust arousal channel from the SDK heart rate. weight is (1 - Q):
    HR carries the verdict when motion corrupts the optical signal and HRV quality
    drops. Confidence is a fixed constant when HR is meaningful, else 0 (no-op)."""
    sdk_hr = ctx.sensor.heart_rate_bpm if ctx.sensor else 0.0
    hr_z = ctx.baseline.hr_z_score(sdk_hr) if ctx.baseline.is_ready else 0.0
    hr_present = sdk_hr > 0.0 and ctx.baseline.is_ready
    hr_conf = HR_CHANNEL_CONFIDENCE if hr_present else 0.0
    return FusionChannel(
        "hr", z=hr_z, weight=1.0 - ctx.quality.quality, confidence=hr_conf
    )


def temp_channel(ctx: ChannelContext) -> FusionChannel | None:
    """Peripheral skin-temperature channel. Built only when the frame carries a
    skin-temperature reading (SKIN_TEMP capability). None when it contributes no
    weight (baseline not ready / inverted, etc.)."""
    if ctx.temperature is None or ctx.sensor is None:
        return None
    ev = evaluate_skin_temperature(
        ctx.temperature,
        skin_temp_c=ctx.sensor.skin_temperature_c,
        ambient_temp_c=ctx.sensor.ambient_temperature_c,
    )
    weight = ev.confidence * TEMP_CHANNEL_MAX_WEIGHT
    if weight <= 0.0:
        return None
    return FusionChannel("temp", z=ev.z, weight=weight, confidence=ev.confidence)


def build_channels(ctx: ChannelContext) -> list[FusionChannel]:
    """Assemble the fusion channels for this epoch, one gate per signal the device
    carries. For a wrist frame (IBI + HR + SKIN_TEMP) this is exactly [hrv, hr, temp]
   , identical to the previous registry output. HRV requires IBI; the pipeline only
    calls decide() once a primary HRV window exists, so IBI is always present here,
    but the gate is kept explicit for a future IBI-less device."""
    channels: list[FusionChannel] = []
    if Capability.IBI in ctx.present:
        channels.append(hrv_channel(ctx))
    if Capability.HR in ctx.present:
        channels.append(hr_channel(ctx))
    if Capability.SKIN_TEMP in ctx.present:
        temp = temp_channel(ctx)
        if temp is not None:
            channels.append(temp)
    return channels
