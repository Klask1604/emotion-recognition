"""Multi-channel decision confidence: the verdict must stay confident in motion.

When the wrist moves, the HRV signal quality (Q) collapses, but the HR channel
still carries the verdict, so the *reported* confidence must floor near the HR
channel confidence (not drop to ~0) and dominant_channel must say "hr". At rest
with a clean signal it is HRV-driven instead.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from biofizic.compute_features.results import HrvMetrics, MultiWindowHrvResult
from biofizic.config import (
    BASELINE_MIN_REST_SAMPLES,
    HR_CHANNEL_CONFIDENCE,
)
from biofizic.engine.decision import DecisionState, decide
from biofizic.engine.pipeline import PhysiologyPipeline
from biofizic.engine.signal_quality import SignalQuality
from biofizic.ingestion.messages import SensorBatchMessage


def _metrics(rmssd: float = 40.0, si: float = 12.0, hr: float = 80.0) -> HrvMetrics:
    return HrvMetrics(
        rmssd_ms=rmssd,
        sdnn_ms=50.0,
        mean_interbeat_interval_ms=60000.0 / hr,
        mean_heart_rate_bpm=hr,
        pnn50_percent=10.0,
        beat_count=40,
        covered_seconds=30.0,
        kubios_stress_index=si,
    )


def _ready_pipeline(tmp_path: Path) -> PhysiologyPipeline:
    p = PhysiologyPipeline()
    p.baseline = type(p.baseline)(path=tmp_path / "rest_baseline.json")
    for hr in (62, 64, 66, 68, 63, 67, 65, 64) * 3:
        p.baseline.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0, heart_rate_bpm=hr)
    assert p.baseline.is_ready
    return p


def _decide(pipeline: PhysiologyPipeline, quality: SignalQuality, sdk_hr: float):
    # Window metrics always have a valid HR; sdk_hr is the (separate) SDK channel
    # whose absence (0) means there is no robust motion channel.
    metrics = _metrics(hr=80.0)
    multi = MultiWindowHrvResult(None, metrics, None, None)
    sensor = SensorBatchMessage(timestamp_ms=0, heart_rate_bpm=sdk_hr)
    return decide(
        primary=metrics,
        multi=multi,
        sensor=sensor,
        quality=quality,
        baseline=pipeline.baseline,
        state=pipeline.decision_state,
        publish_epoch=True,
    )


def test_moving_confidence_floors_on_hr_channel(tmp_path: Path):
    pipeline = _ready_pipeline(tmp_path)
    moving_q = SignalQuality(
        quality=0.03,  # HRV quality collapsed by motion
        usable=False,
        artifact_rate=0.1,
        motion_energy=5.0,
        p_artifact=0.5,
        motion_state="moving",
    )
    d = _decide(pipeline, moving_q, sdk_hr=110.0)
    # Verdict is HR-driven and stays confident, not ~3%.
    assert d.dominant_channel == "hr"
    assert d.decision_confidence >= 0.5
    assert d.decision_confidence == pytest.approx(
        0.03 * 0.03 + 0.97 * HR_CHANNEL_CONFIDENCE, abs=1e-6
    )
    # Diagnostic HRV-only quality is still exposed low (honest).
    assert d.signal_quality == pytest.approx(0.03)


def test_still_confidence_is_hrv_driven(tmp_path: Path):
    pipeline = _ready_pipeline(tmp_path)
    still_q = SignalQuality(
        quality=0.9,
        usable=True,
        artifact_rate=0.0,
        motion_energy=0.0,
        p_artifact=0.05,
        motion_state="still",
    )
    d = _decide(pipeline, still_q, sdk_hr=66.0)
    assert d.dominant_channel == "hrv"
    assert d.decision_confidence >= 0.8


def test_no_hr_reports_none_channel(tmp_path: Path):
    pipeline = _ready_pipeline(tmp_path)
    moving_q = SignalQuality(
        quality=0.03,
        usable=False,
        artifact_rate=0.1,
        motion_energy=5.0,
        p_artifact=0.5,
        motion_state="moving",
    )
    d = _decide(pipeline, moving_q, sdk_hr=0.0)  # SDK gives no HR
    assert d.dominant_channel == "none"
    # Without a robust channel, confidence does not get the HR floor.
    assert d.decision_confidence < 0.1


def _cold_pipeline(tmp_path: Path) -> PhysiologyPipeline:
    """A pipeline whose baseline has NOT been locked yet (no resting epochs)."""
    p = PhysiologyPipeline()
    p.baseline = type(p.baseline)(path=tmp_path / "rest_baseline.json")
    assert not p.baseline.is_ready
    return p


def test_preliminary_fidelity_pre_baseline(tmp_path: Path):
    """Before baseline lock the verdict comes from Kubios population zones —
    decision must declare itself preliminary so the UI can show a badge."""
    from biofizic.config import PRELIMINARY_CONFIDENCE_CAP

    pipeline = _cold_pipeline(tmp_path)
    still_q = SignalQuality(
        quality=0.9,
        usable=True,
        artifact_rate=0.0,
        motion_energy=0.0,
        p_artifact=0.05,
        motion_state="still",
    )
    d = _decide(pipeline, still_q, sdk_hr=66.0)
    assert d.decision_fidelity == "preliminary"
    # Confidence must be capped so a preliminary verdict cannot look like a
    # calibrated one in the UI (was: 0.9 of clean HRV → looked fully calibrated).
    assert d.decision_confidence <= PRELIMINARY_CONFIDENCE_CAP + 1e-6
    # Baseline label is still "pending" — that's the existing convention.
    assert d.baseline_label == "pending"


def test_calibrated_fidelity_after_baseline(tmp_path: Path):
    """Once the personal baseline locks, fidelity flips to calibrated and the
    confidence cap is lifted (high-q still epoch can reach ~0.9)."""
    pipeline = _ready_pipeline(tmp_path)
    still_q = SignalQuality(
        quality=0.9,
        usable=True,
        artifact_rate=0.0,
        motion_energy=0.0,
        p_artifact=0.05,
        motion_state="still",
    )
    d = _decide(pipeline, still_q, sdk_hr=66.0)
    assert d.decision_fidelity == "calibrated"
    assert d.decision_confidence >= 0.8  # cap no longer applies
