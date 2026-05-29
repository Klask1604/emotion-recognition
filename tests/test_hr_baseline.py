"""Personal HR baseline + hr_z (the motion-robust fusion channel)."""

from __future__ import annotations

from pathlib import Path

import pytest

from biofizic.engine.baseline import RestBaselineStore
from biofizic.config import BASELINE_MIN_REST_SAMPLES


@pytest.fixture
def store(tmp_path: Path) -> RestBaselineStore:
    s = RestBaselineStore(path=tmp_path / "rest_baseline.json")
    for _ in range(BASELINE_MIN_REST_SAMPLES):
        s.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0, heart_rate_bpm=65.0)
    return s


def test_hr_baseline_locks_and_zero_at_rest(store: RestBaselineStore):
    assert store.is_ready
    assert store.baseline_heart_rate_bpm == pytest.approx(65.0, rel=1e-6)
    assert store.hr_z_score(65.0) == pytest.approx(0.0, abs=1e-6)


def test_hr_z_positive_when_elevated(store: RestBaselineStore):
    # Need spread for a finite sigma; feed a small resting range.
    s = store
    for hr in (62, 64, 66, 68, 63, 67, 65, 64):
        s.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0, heart_rate_bpm=hr)
    assert s.hr_z_score(110.0) > 0  # exertion HR well above resting
    assert s.hr_z_score(60.0) < 0


def test_hr_baseline_persists(tmp_path: Path):
    p = tmp_path / "rest_baseline.json"
    s = RestBaselineStore(path=p)
    for _ in range(BASELINE_MIN_REST_SAMPLES):
        s.observe_resting(rmssd_ms=50.0, kubios_stress_index=12.0, heart_rate_bpm=70.0)
    reloaded = RestBaselineStore(path=p)
    assert reloaded.baseline_heart_rate_bpm == pytest.approx(s.baseline_heart_rate_bpm)
