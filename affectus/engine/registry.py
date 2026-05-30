"""
Channel registry: capability-gated assembly of the fusion channels.

Each fusion channel declares which Capabilities it needs; the registry builds a
channel only when those capabilities are present in the current frame/context.
This generalises the hand-written channel list previously inline in
decision.decide() and the conditional engine construction in legacy.__init__.

For a wrist frame (IBI + HR + SKIN_TEMP) this produces exactly the prior list
[hrv, hr, temp], so the fused verdict is unchanged. A device without SKIN_TEMP
simply never builds the temp channel (not a weight-0 channel — it is skipped).

Two clean gates, kept separate:
  - capability present?  -> build/evaluate the channel at all
  - weight > 0?          -> the channel actually contributes (existing math)
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.config import HR_CHANNEL_CONFIDENCE, TEMP_CHANNEL_MAX_WEIGHT
from affectus.engine.channels.temperature import evaluate_skin_temperature
from affectus.engine.decision import FusionChannel
from affectus.sensing.capabilities import Capability


@dataclass
class ChannelContext:
    """Everything a channel evaluator may need for one epoch. Assembled by the
    pipeline; channels read only what their capability grants."""

    primary: object                      # HrvMetrics
    sensor: object | None                # SensorBatchMessage
    quality: object                      # SignalQuality
    baseline: object                     # RestBaselineStore
    temperature: object | None = None    # SkinTemperatureChannelState
    valence_wesad: object | None = None  # ValenceWesadState
    ppg_green: list | None = None        # raw PPG window (for valence features)
    ppg_ts_ms: list | None = None
    present: frozenset[Capability] = frozenset()


# ── Per-channel evaluators (return a FusionChannel, or None when absent/zero) ──

def _eval_hrv(ctx: ChannelContext) -> FusionChannel | None:
    stress_z = (
        ctx.baseline.stress_index_z_score(ctx.primary.kubios_stress_index)
        if ctx.baseline.is_ready else 0.0
    )
    return FusionChannel(
        "hrv", z=stress_z, weight=ctx.quality.quality, confidence=ctx.quality.quality
    )


def _eval_hr(ctx: ChannelContext) -> FusionChannel | None:
    sdk_hr = ctx.sensor.heart_rate_bpm if ctx.sensor else 0.0
    hr_z = ctx.baseline.hr_z_score(sdk_hr) if ctx.baseline.is_ready else 0.0
    hr_present = sdk_hr > 0.0 and ctx.baseline.is_ready
    hr_conf = HR_CHANNEL_CONFIDENCE if hr_present else 0.0
    return FusionChannel(
        "hr", z=hr_z, weight=1.0 - ctx.quality.quality, confidence=hr_conf
    )


def _eval_temp(ctx: ChannelContext) -> FusionChannel | None:
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


def _eval_valence_wesad(ctx: ChannelContext) -> FusionChannel | None:
    """WESAD-trained valence channel. Predicts positive/negative valence from the
    raw PPG window and contributes a capped, confidence-weighted z. Skipped
    entirely when the model is absent, PPG is missing, or the cap is 0."""
    if ctx.valence_wesad is None or not ctx.ppg_green:
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
    return FusionChannel("valence_wesad", z=res.z, weight=weight, confidence=res.confidence)


@dataclass(frozen=True)
class ChannelSpec:
    name: str
    requires: frozenset[Capability]
    evaluate: object   # Callable[[ChannelContext], FusionChannel | None]


# Order matters only for diagnostics; _fuse is order-independent.
CHANNEL_REGISTRY: list[ChannelSpec] = [
    ChannelSpec("hrv", frozenset({Capability.IBI}), _eval_hrv),
    ChannelSpec("hr", frozenset({Capability.HR}), _eval_hr),
    ChannelSpec("temp", frozenset({Capability.SKIN_TEMP}), _eval_temp),
    # Valence model needs the raw PPG waveform (vascular/morphology features).
    # Default-disabled via VALENCE_WESAD_MAX_WEIGHT=0 -> evaluate returns None.
    ChannelSpec("valence_wesad",
                frozenset({Capability.IBI, Capability.PPG}), _eval_valence_wesad),
    # Future: ChannelSpec("eda", frozenset({Capability.EDA}), _eval_eda)
]


def build_channels(ctx: ChannelContext) -> list[FusionChannel]:
    """Run every registered channel whose required capabilities are present and
    that returns a non-None contribution."""
    out: list[FusionChannel] = []
    for spec in CHANNEL_REGISTRY:
        if not (spec.requires <= ctx.present):
            continue
        ch = spec.evaluate(ctx)
        if ch is not None:
            out.append(ch)
    return out
