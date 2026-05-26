"""IBI filtering and successive-difference math."""

from __future__ import annotations

from biofizic.signal import (
    filter_physiological_intervals,
    successive_interval_differences,
)
from biofizic.types import InterbeatIntervalEntry


def _entries(values: list[int]) -> list[InterbeatIntervalEntry]:
    return [InterbeatIntervalEntry(interval_ms=v) for v in values]


def test_filter_drops_non_physiological_intervals():
    entries = _entries([280, 800, 820, 2100, 810])
    out = filter_physiological_intervals(entries)
    kept = [e.interval_ms for e in out]
    assert 280 not in kept, "values below 300 ms must be rejected"
    assert 2100 not in kept, "values above 2000 ms must be rejected"
    assert kept == [800, 820, 810]


def test_filter_drops_median_outliers_beyond_20_percent():
    # median is 800; 20 percent of 800 is 160; 1200 is 400 above so must drop
    entries = _entries([790, 800, 810, 1200])
    out = filter_physiological_intervals(entries)
    assert 1200 not in [e.interval_ms for e in out]


def test_successive_differences_use_consecutive_delta_when_no_timestamps():
    entries = _entries([800, 820, 810, 830])
    diffs = successive_interval_differences(entries)
    assert diffs == [20.0, -10.0, 20.0]


def test_successive_differences_prefer_timestamp_coherent_pairs():
    # Adjacent IBIs with timestamps that match the interval delta within
    # the mismatch tolerance should use the timestamp-coherent branch.
    entries = [
        InterbeatIntervalEntry(interval_ms=800, timestamp_ms=10_000),
        InterbeatIntervalEntry(interval_ms=820, timestamp_ms=10_820),
        InterbeatIntervalEntry(interval_ms=810, timestamp_ms=11_630),
    ]
    diffs = successive_interval_differences(entries)
    assert diffs == [20.0, -10.0]
