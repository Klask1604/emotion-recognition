"""Personal resting baseline: lock-in, EMA updates, recalibration reset."""

from __future__ import annotations

from pathlib import Path

import pytest

from biofizic.baseline import RestBaselineStore
from biofizic.config import STILL_EPOCHS_BEFORE_BASELINE_LOCK


@pytest.fixture
def baseline_path(tmp_path: Path) -> Path:
    return tmp_path / "rest_baseline.json"


def test_baseline_locks_after_required_still_observations(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    assert not store.is_ready
    for _ in range(STILL_EPOCHS_BEFORE_BASELINE_LOCK):
        store.observe_still(rmssd_ms=45.0, kubios_stress_index=10.0)
    assert store.is_ready
    assert store.baseline_rmssd_ms is not None
    assert store.baseline_stress_index is not None


def test_baseline_persists_to_disk_and_reloads(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    for _ in range(STILL_EPOCHS_BEFORE_BASELINE_LOCK):
        store.observe_still(rmssd_ms=50.0, kubios_stress_index=12.0)
    assert baseline_path.exists()

    reloaded = RestBaselineStore(path=baseline_path)
    assert reloaded.is_ready
    assert reloaded.baseline_rmssd_ms == store.baseline_rmssd_ms
    assert reloaded.baseline_stress_index == store.baseline_stress_index


def test_reset_for_recalibration_clears_state(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    for _ in range(STILL_EPOCHS_BEFORE_BASELINE_LOCK):
        store.observe_still(rmssd_ms=45.0, kubios_stress_index=10.0)
    assert store.is_ready

    store.reset_for_recalibration()
    assert not store.is_ready
    assert store.baseline_rmssd_ms is None
    assert store.baseline_stress_index is None
    assert store.still_observation_count == 0
