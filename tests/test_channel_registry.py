"""Capability-gated channel registry: channels build only when their required
capabilities are present, and the wrist set reproduces the prior [hrv,hr,temp]."""

from __future__ import annotations

from affectus.engine.registry import ChannelContext, build_channels
from affectus.sensing.capabilities import Capability
from affectus.engine.signal_quality import SignalQuality
from affectus.engine.channels.temperature import SkinTemperatureChannelState
from affectus.engine.baseline import RestBaselineStore
from affectus.compute_features.results import HrvMetrics
from affectus.ingestion.messages import SensorBatchMessage


def _quality(q=0.9):
    return SignalQuality(quality=q, usable=True, artifact_rate=0.0,
                         motion_energy=0.0, p_artifact=0.0, motion_state="still")


def _metrics():
    return HrvMetrics(rmssd_ms=40, sdnn_ms=50, mean_interbeat_interval_ms=800,
                      mean_heart_rate_bpm=75, pnn50_percent=10, beat_count=40,
                      covered_seconds=30, kubios_stress_index=10.0)


def _ctx(present, temperature=None, tmp_path=None):
    return ChannelContext(
        primary=_metrics(),
        sensor=SensorBatchMessage(timestamp_ms=0, heart_rate_bpm=70.0,
                                  skin_temperature_c=33.0, ambient_temperature_c=24.0),
        quality=_quality(),
        baseline=RestBaselineStore(path=(tmp_path / "b.json") if tmp_path else None),
        temperature=temperature,
        present=frozenset(present),
    )


def test_wrist_set_yields_hrv_and_hr():
    chans = build_channels(_ctx({Capability.IBI, Capability.HR}))
    names = {c.name for c in chans}
    assert names == {"hrv", "hr"}   # no SKIN_TEMP capability -> no temp channel


def test_no_hr_capability_skips_hr_channel():
    chans = build_channels(_ctx({Capability.IBI}))
    assert {c.name for c in chans} == {"hrv"}


def test_temp_capability_without_locked_baseline_is_skipped_not_zero():
    # SKIN_TEMP present + temperature state, but baseline not locked -> evaluate
    # returns weight 0 -> channel skipped entirely (not a weight-0 entry).
    temp = SkinTemperatureChannelState()   # unlocked
    chans = build_channels(_ctx({Capability.IBI, Capability.HR, Capability.SKIN_TEMP},
                                temperature=temp))
    assert "temp" not in {c.name for c in chans}


def test_temp_capability_absent_means_no_temp_even_with_state(tmp_path):
    # Even if a temperature state object exists, without the SKIN_TEMP capability
    # the channel is never built (capability gate is separate from weight).
    temp = SkinTemperatureChannelState()
    chans = build_channels(_ctx({Capability.IBI, Capability.HR}, temperature=temp,
                                tmp_path=tmp_path))
    assert "temp" not in {c.name for c in chans}
