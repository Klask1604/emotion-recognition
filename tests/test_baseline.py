"""Robust personal baseline: lock-in, log-space z-scores, recalibration reset."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from biofizic.engine.baseline import RestBaselineStore
from biofizic.config import BASELINE_MIN_REST_EPOCHS


@pytest.fixture
def baseline_path(tmp_path: Path) -> Path:
    return tmp_path / "rest_baseline.json"


def test_baseline_locks_after_required_rest_observations(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    assert not store.is_ready
    for _ in range(BASELINE_MIN_REST_EPOCHS):
        store.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0)
    assert store.is_ready
    assert store.baseline_rmssd_ms is not None
    assert store.baseline_stress_index is not None
    # Median of constant input recovers the input value.
    assert store.baseline_stress_index == pytest.approx(10.0, rel=1e-6)


def test_zscore_is_zero_at_baseline_and_signed_correctly(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    # Spread of resting SI so MAD (sigma) is non-degenerate.
    for si in [9.0, 10.0, 11.0, 10.5, 9.5, 10.0, 11.5, 8.5, 10.2, 9.8, 10.1, 9.9]:
        store.observe_resting(rmssd_ms=45.0, kubios_stress_index=si)
    assert store.is_ready
    median_si = store.baseline_stress_index
    assert store.stress_index_z_score(median_si) == pytest.approx(0.0, abs=1e-6)
    # SI above baseline -> positive stress z; RMSSD below baseline -> positive z.
    assert store.stress_index_z_score(median_si * 1.5) > 0
    assert store.rmssd_z_score(20.0) > 0


def test_baseline_persists_to_disk_and_reloads(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    for _ in range(BASELINE_MIN_REST_EPOCHS):
        store.observe_resting(rmssd_ms=50.0, kubios_stress_index=12.0)
    assert baseline_path.exists()

    reloaded = RestBaselineStore(path=baseline_path)
    assert reloaded.is_ready
    assert reloaded.baseline_rmssd_ms == pytest.approx(store.baseline_rmssd_ms)
    assert reloaded.baseline_stress_index == pytest.approx(store.baseline_stress_index)


def test_reset_for_recalibration_clears_state(baseline_path: Path):
    store = RestBaselineStore(path=baseline_path)
    for _ in range(BASELINE_MIN_REST_EPOCHS):
        store.observe_resting(rmssd_ms=45.0, kubios_stress_index=10.0)
    assert store.is_ready

    store.reset_for_recalibration()
    assert not store.is_ready
    assert store.baseline_rmssd_ms is None
    assert store.baseline_stress_index is None
    assert store.rest_observation_count == 0
