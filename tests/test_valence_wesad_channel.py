"""The WESAD valence fusion channel: predicts from PPG, contributes only a
capped weight, and is an exact no-op when disabled (cap 0) or model/PPG absent."""

from __future__ import annotations

import math

import pytest

from affectus.engine.channels.valence_wesad import (
    ValenceWesadState,
    evaluate_valence_wesad,
    valence_weight,
)


def _synth_ppg(seconds=20, fs=64):
    g = [int(2000 + 500 * math.sin(2 * math.pi * 1.4 * i / fs)) for i in range(seconds * fs)]
    ts = [int(i * 1000 / fs) for i in range(seconds * fs)]
    return g, ts


def test_predicts_when_model_present():
    state = ValenceWesadState()
    state.ensure_loaded()
    if state.model is None:
        pytest.skip("model not trained (models/valence_wesad.joblib missing)")
    g, ts = _synth_ppg()
    res = evaluate_valence_wesad(state, g, ts)
    assert res is not None
    assert -1.0 <= res.z <= 1.0
    assert 0.0 <= res.p_positive <= 1.0
    assert res.confidence == abs(res.z)


def test_no_model_returns_none():
    state = ValenceWesadState(_loaded=True)  # pretend loaded, but model is None
    g, ts = _synth_ppg()
    assert evaluate_valence_wesad(state, g, ts) is None


def test_weight_is_zero_by_default_disabled():
    # VALENCE_WESAD_MAX_WEIGHT defaults to 0 -> the channel never contributes,
    # so the production verdict is unchanged until the cap is explicitly raised.
    assert valence_weight(1.0) == 0.0


def test_channel_skipped_in_registry_when_disabled(tmp_path):
    from affectus.engine.registry import ChannelContext, build_channels
    from affectus.sensing.capabilities import Capability
    from affectus.engine.signal_quality import SignalQuality
    from affectus.engine.baseline import RestBaselineStore
    from affectus.compute_features.results import HrvMetrics
    from affectus.ingestion.messages import SensorBatchMessage

    state = ValenceWesadState()
    g, ts = _synth_ppg()
    ctx = ChannelContext(
        primary=HrvMetrics(rmssd_ms=40, sdnn_ms=50, mean_interbeat_interval_ms=800,
                           mean_heart_rate_bpm=75, pnn50_percent=10, beat_count=40,
                           covered_seconds=30, kubios_stress_index=10.0),
        sensor=SensorBatchMessage(timestamp_ms=0, heart_rate_bpm=70.0),
        quality=SignalQuality(quality=0.9, usable=True, artifact_rate=0.0,
                              motion_energy=0.0, p_artifact=0.0, motion_state="still"),
        baseline=RestBaselineStore(path=tmp_path / "b.json"),
        valence_wesad=state, ppg_green=g, ppg_ts_ms=ts,
        present=frozenset({Capability.IBI, Capability.HR, Capability.PPG}),
    )
    names = {c.name for c in build_channels(ctx)}
    # cap is 0 -> valence channel produces no contribution -> not in the list
    assert "valence_wesad" not in names
    assert {"hrv", "hr"} <= names
