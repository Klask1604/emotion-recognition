"""Filter and parse inter-beat interval streams."""

from __future__ import annotations

from biofizic.config import (
    MAX_INTERBEAT_INTERVAL_MS,
    MAX_TIMESTAMP_IBI_MISMATCH_MS,
    MIN_INTERBEAT_INTERVAL_MS,
    OUTLIER_MEDIAN_DEVIATION_RATIO,
)
from biofizic.types import InterbeatIntervalEntry


def parse_intervals_from_payload(data: dict) -> list[InterbeatIntervalEntry]:
    """Parse ibi_ms and ibi_ts arrays from MQTT JSON."""
    intervals = [int(x) for x in (data.get("ibi_ms") or []) if int(x) > 0]
    if not intervals:
        return []
    raw_ts = data.get("ibi_ts") or data.get("ibi_ts_ms") or []
    if isinstance(raw_ts, list) and len(raw_ts) == len(intervals):
        return [
            InterbeatIntervalEntry(interval_ms=ms, timestamp_ms=int(ts))
            for ms, ts in zip(intervals, raw_ts)
        ]
    return [InterbeatIntervalEntry(interval_ms=ms) for ms in intervals]


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
    RMSSD input: prefer timestamp-coherent pairs, else simple consecutive delta.
    """
    diffs: list[float] = []
    for i in range(len(entries) - 1):
        left = entries[i]
        right = entries[i + 1]
        if left.timestamp_ms is not None and right.timestamp_ms is not None:
            gap_ms = right.timestamp_ms - left.timestamp_ms
            if abs(gap_ms - right.interval_ms) < MAX_TIMESTAMP_IBI_MISMATCH_MS:
                diffs.append(float(right.interval_ms - left.interval_ms))
                continue
        diffs.append(float(right.interval_ms - left.interval_ms))
    return diffs


def trim_entries_to_lookback(
    entries: list[InterbeatIntervalEntry],
    *,
    end_timestamp_ms: int | None,
    max_span_ms: int,
) -> list[InterbeatIntervalEntry]:
    """Keep only entries within max_span_ms before end_timestamp_ms."""
    if not entries or not any(e.timestamp_ms is not None for e in entries):
        return entries
    end = end_timestamp_ms
    if end is None or end <= 0:
        timestamps = [e.timestamp_ms for e in entries if e.timestamp_ms is not None]
        end = max(timestamps) if timestamps else 0
    if end <= 0:
        return entries
    cutoff = end - max_span_ms
    trimmed = [e for e in entries if e.timestamp_ms is not None and e.timestamp_ms >= cutoff]
    return trimmed if len(trimmed) >= 2 else entries
