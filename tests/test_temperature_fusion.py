"""B1b integration: the skin-temperature channel feeding the decision fusion.

Pins three contracts:
  1. With no temperature state (temperature=None) the verdict is identical to
     the HRV+HR-only pipeline — exact backward compatibility.
  2. A locked temperature baseline + a colder-than-baseline reading nudges the
     fused arousal UP (sympathetic vasoconstriction), but
  3. only within the conservative weight cap — temperature cannot by itself
     swamp a clean HRV/HR verdict.
"""

from __future__ import annotations

from pathlib import Path

from biofizic.compute_features.results import HrvMetrics, MultiWindowHrvResult
from biofizic.config import (
    BASELINE_MIN_REST_SAMPLES,
    TEMP_BASELINE_MIN_REST_EPOCHS,
)
from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.channels.temperature import SkinTemperatureChannelState
from biofizic.engine.decision import DecisionState, decide
from biofizic.engine.signal_quality import SignalQuality
from biofizic.ingestion.messages import SensorBatchMessage


def _metrics(rmssd: float = 35.0, si: float = 18.0, hr: float = 85.0) -> HrvMetrics:
    return HrvMetrics(
        rmssd_ms=rmssd, sdnn_ms=45.0, mean_interbeat_interval_ms=60000.0 / hr,
        mean_heart_rate_bpm=hr, pnn50_percent=8.0, beat_count=60,
        covered_seconds=60.0, kubios_stress_index=si,
    )


def _ready_hrv_baseline(tmp_path: Path) -> RestBaselineStore:
    b = RestBaselineStore(path=tmp_path / "rest.json")
    # Spread so MAD is non-degenerate; no `now` => spacing gate off.
    for si in [9.0, 10.0, 11.0, 10.5, 9.5, 10.2, 11.5, 8.5, 10.1, 9.9, 10.0, 10.3]:
        b.observe_resting(rmssd_ms=45.0, kubios_stress_index=si, heart_rate_bpm=65.0)
    assert b.is_ready
    return b


def _locked_temp(skin_c: float = 33.0, ambient_c: float = 24.0) -> SkinTemperatureChannelState:
    s = SkinTemperatureChannelState()  # in-memory, no persistence
    for i in range(TEMP_BASELINE_MIN_REST_EPOCHS):
        s.observe_resting(skin_temp_c=skin_c + (0.1 if i % 2 else -0.1), ambient_temp_c=ambient_c)
    assert s.is_ready
    return s


def _still_quality() -> SignalQuality:
    return SignalQuality(
        quality=0.9, usable=True, artifact_rate=0.0, motion_energy=0.0,
        p_artifact=0.0, motion_state="still",
    )


def _decide(baseline, temperature, *, skin_c: float, ambient_c: float = 24.0):
    metrics = _metrics()
    multi = MultiWindowHrvResult(None, metrics, None, None)
    sensor = SensorBatchMessage(
        timestamp_ms=0, heart_rate_bpm=85.0,
        skin_temperature_c=skin_c, ambient_temperature_c=ambient_c,
    )
    return decide(
        primary=metrics, multi=multi, sensor=sensor, quality=_still_quality(),
        baseline=baseline, temperature=temperature, state=DecisionState(),
        publish_epoch=True,
    )


def test_no_temperature_state_is_identical(tmp_path: Path):
    """temperature=None must reproduce the HRV+HR-only verdict exactly."""
    b1 = _ready_hrv_baseline(tmp_path)
    d_none = _decide(b1, None, skin_c=30.0)

    b2 = _ready_hrv_baseline(tmp_path / "b2")
    # Even with a temperature object, an UNLOCKED one contributes weight 0.
    cold_unlocked = SkinTemperatureChannelState()
    d_unlocked = _decide(b2, cold_unlocked, skin_c=30.0)

    assert d_none.stress_index_z_filtered == d_unlocked.stress_index_z_filtered
    assert d_none.display_arousal_10 == d_unlocked.display_arousal_10


def test_colder_skin_nudges_arousal_up(tmp_path: Path):
    """A locked temp baseline + colder reading raises the fused z vs a neutral
    (at-baseline) reading."""
    baseline = _ready_hrv_baseline(tmp_path)
    temp = _locked_temp(skin_c=33.0)

    at_baseline = _decide(baseline, temp, skin_c=33.0)  # temp z ~ 0
    # Fresh decision state for an independent comparison.
    baseline2 = _ready_hrv_baseline(tmp_path / "b2")
    temp2 = _locked_temp(skin_c=33.0)
    colder = _decide(baseline2, temp2, skin_c=31.5)  # below baseline -> arousal

    assert colder.stress_index_z_filtered > at_baseline.stress_index_z_filtered


def test_temperature_cannot_dominate(tmp_path: Path):
    """Even an extreme cold reading moves the fused z by no more than the cap
    allows relative to the no-temp verdict — temperature only nudges."""
    baseline = _ready_hrv_baseline(tmp_path)
    no_temp = _decide(baseline, None, skin_c=31.0)

    baseline2 = _ready_hrv_baseline(tmp_path / "b2")
    temp = _locked_temp(skin_c=33.0)
    extreme_cold = _decide(baseline2, temp, skin_c=28.0)  # way below baseline

    # The temperature push is bounded; it must not flip the verdict wildly.
    # (Sanity ceiling: arousal stays within a few points of the no-temp value.)
    assert abs(extreme_cold.display_arousal_10 - no_temp.display_arousal_10) <= 3
