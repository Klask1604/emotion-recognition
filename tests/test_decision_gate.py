"""Decision gate (population vs personal arousal + CUSUM alert).

The gate logic now lives in `biofizic.engine.decision.decide`. These tests
drive `decide()` directly with synthetic HrvMetrics / SignalQuality / baseline
states to cover the same behaviours that used to live in `decision_gate.py`.
"""

from __future__ import annotations

from pathlib import Path

from biofizic.compute_features.results import HrvMetrics, MultiWindowHrvResult
from biofizic.config import CUSUM_THRESHOLD_H
from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.decision import DecisionState, _cusum_update, decide
from biofizic.engine.signal_quality import SignalQuality
from biofizic.ingestion.messages import SensorBatchMessage


def _quality(*, q: float = 0.9, motion: str = "still", artifact: float = 0.0) -> SignalQuality:
    return SignalQuality(
        quality=q,
        usable=artifact <= 0.05,
        artifact_rate=artifact,
        motion_energy=0.0,
        p_artifact=0.0,
        motion_state=motion,
    )


def _metrics(si: float = 10.0, rmssd: float = 40.0, hr: float = 70.0) -> HrvMetrics:
    return HrvMetrics(
        rmssd_ms=rmssd,
        sdnn_ms=50.0,
        mean_interbeat_interval_ms=60_000.0 / hr,
        mean_heart_rate_bpm=hr,
        pnn50_percent=10.0,
        beat_count=40,
        covered_seconds=30.0,
        kubios_stress_index=si,
    )


def _cold_baseline(tmp_path: Path) -> RestBaselineStore:
    return RestBaselineStore(path=tmp_path / "rest_baseline.json")


def _ready_baseline(tmp_path: Path) -> RestBaselineStore:
    b = RestBaselineStore(path=tmp_path / "rest_baseline.json")
    for hr in (62, 64, 66, 68, 63, 67, 65, 64) * 3:
        b.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0, heart_rate_bpm=hr)
    assert b.is_ready
    return b


def _decide_with(
    *,
    si: float,
    z_filtered_seed: float = 0.0,
    baseline: RestBaselineStore,
    state: DecisionState | None = None,
) -> object:
    """Drive decide() once with a synthetic metric. Uses publish_epoch=True so
    the Kalman folds in; we pre-seed the estimator x when needed to mimic a
    'filtered z is already at X' starting condition."""
    s = state or DecisionState()
    if z_filtered_seed != 0.0:
        s.estimator_x = z_filtered_seed
    primary = _metrics(si=si)
    multi = MultiWindowHrvResult(None, primary, None, None)
    sensor = SensorBatchMessage(timestamp_ms=0, heart_rate_bpm=70.0)
    return decide(
        primary=primary,
        multi=multi,
        sensor=sensor,
        quality=_quality(),
        baseline=baseline,
        state=s,
        publish_epoch=True,
    )


def test_pre_baseline_uses_population_zone(tmp_path: Path):
    """Without a personal baseline, arousal_10 comes from the Kubios population
    zones. SI=10 falls in NORMAL → arousal 5, label 'Echilibrat'."""
    d = _decide_with(si=10.0, baseline=_cold_baseline(tmp_path))
    assert d.display_arousal_10 == 5
    assert d.kubios_label == "Echilibrat"
    assert d.decision_fidelity == "preliminary"


def test_post_baseline_uses_filtered_z(tmp_path: Path):
    """Once the personal baseline is locked, arousal comes from the filtered z
    CDF. A strongly positive filtered z (way above resting) → high arousal."""
    state = DecisionState(estimator_x=3.0)
    baseline = _ready_baseline(tmp_path)
    primary = _metrics(si=10.0)
    multi = MultiWindowHrvResult(None, primary, None, None)
    sensor = SensorBatchMessage(timestamp_ms=0, heart_rate_bpm=70.0)
    d = decide(
        primary=primary,
        multi=multi,
        sensor=sensor,
        quality=_quality(),
        baseline=baseline,
        state=state,
        publish_epoch=False,  # do NOT update Kalman, keep seeded x
    )
    assert d.decision_fidelity == "calibrated"
    assert d.display_arousal_10 == 10  # Φ(3 + small offset) ≈ 0.998 → round(1+9·1)=10


def test_cusum_confirms_sustained_elevation():
    """CUSUM accumulates over repeated z above the slack k; once S > h it latches.
    Drive _cusum_update directly to keep the test focused on the detector."""
    state = DecisionState()
    fired = any(_cusum_update(state, 2.0) for _ in range(20))
    assert fired
    assert state.cusum_s > CUSUM_THRESHOLD_H


def test_cusum_quiet_under_noise():
    """Centered noise should NOT accumulate the CUSUM past the threshold."""
    state = DecisionState()
    for i in range(50):
        z = 0.3 if i % 2 == 0 else -0.3
        alert = _cusum_update(state, z)
        assert not alert
