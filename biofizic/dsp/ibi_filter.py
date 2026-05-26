"""Filter inter-beat interval streams for HRV computation."""

from __future__ import annotations

from biofizic.config import (
    MAX_INTERBEAT_INTERVAL_MS,
    MAX_TIMESTAMP_IBI_MISMATCH_MS,
    MIN_INTERBEAT_INTERVAL_MS,
    OUTLIER_MEDIAN_DEVIATION_RATIO,
)
from biofizic.ingestion.messages import InterbeatIntervalEntry


def filter_physiological_intervals(
    entries: list[InterbeatIntervalEntry],
) -> list[InterbeatIntervalEntry]:
    """Keep IBI within 300-2000 ms and remove median outliers (>20%)."""
    physiological = [
        e
        for e in entries
        if MIN_INTERBEAT_INTERVAL_MS <= e.interval_ms <= MAX_INTERBEAT_INTERVAL_MS
    ]
    if len(physiological) < 2:
        return physiological

    sorted_ms = sorted(e.interval_ms for e in physiological)
    median = float(sorted_ms[len(sorted_ms) // 2])
    return [
        e
        for e in physiological
        if abs(e.interval_ms - median) < OUTLIER_MEDIAN_DEVIATION_RATIO * median
    ]


def successive_interval_differences(
    entries: list[InterbeatIntervalEntry],
) -> list[float]:
    """
    RMSSD input: use only timestamp-coherent successive pairs.

    A pair is coherent when the gap between the two beats' timestamps matches
    the later beat's IBI within MAX_TIMESTAMP_IBI_MISMATCH_MS. Pairs that
    straddle a dropped-beat gap are skipped, so RMSSD is not inflated by
    differences taken across discontinuities. This mirrors the on-watch
    HrvFeatureCalculator.successiveDiffsWithTemporalCheck (Android IbiPipeline).

    When no per-beat timestamps are available (or no pair turns out coherent),
    fall back to the plain consecutive deltas.
    """
    coherent: list[float] = []
    for i in range(len(entries) - 1):
        left = entries[i]
        right = entries[i + 1]
        if left.timestamp_ms is None or right.timestamp_ms is None:
            continue
        gap_ms = right.timestamp_ms - left.timestamp_ms
        if abs(gap_ms - right.interval_ms) < MAX_TIMESTAMP_IBI_MISMATCH_MS:
            coherent.append(float(right.interval_ms - left.interval_ms))
    if coherent:
        return coherent
    return [
        float(entries[i + 1].interval_ms - entries[i].interval_ms)
        for i in range(len(entries) - 1)
    ]
