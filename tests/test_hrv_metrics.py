"""HRV math: RMSSD, SDNN, pNN50 and the Baevsky stress index."""

from __future__ import annotations

import math

import numpy as np

from biofizic.compute_features.hrv_metrics import (
    compute_baevsky_indices,
    compute_hrv_from_entries,
)
from biofizic.ingestion.messages import InterbeatIntervalEntry


def test_compute_hrv_returns_none_for_empty_input():
    assert compute_hrv_from_entries([]) is None


def test_compute_hrv_known_intervals():
    # Stable 75 bpm with a small alternating jitter.
    intervals = [800, 820, 810, 830, 805, 825, 815, 800, 810, 820]
    entries = [InterbeatIntervalEntry(interval_ms=v) for v in intervals]
    metrics = compute_hrv_from_entries(entries)
    assert metrics is not None
    assert metrics.beat_count == len(intervals)
    expected_mean = float(np.mean(intervals))
    assert math.isclose(metrics.mean_interbeat_interval_ms, expected_mean, rel_tol=1e-6)
    expected_hr = 60_000.0 / expected_mean
    assert math.isclose(metrics.mean_heart_rate_bpm, expected_hr, rel_tol=1e-6)
    # RMSSD must be positive and small for a near-constant rhythm.
    assert 0 < metrics.rmssd_ms < 30
    assert metrics.kubios_stress_index >= 0.0


def test_kubios_stress_index_is_sqrt_of_baevsky_raw():
    intervals = np.array([800, 820, 810, 830, 805, 825, 815, 800, 810, 820], dtype=float)
    raw, kubios = compute_baevsky_indices(intervals)
    if raw == 0.0:
        # the histogram bin can be empty for tiny inputs; just confirm both 0
        assert kubios == 0.0
    else:
        assert math.isclose(kubios, math.sqrt(raw), rel_tol=1e-6)
