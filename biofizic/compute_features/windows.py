"""Compute HRV metrics for multiple window lengths and rolling buffers."""

from __future__ import annotations

from collections import deque

from biofizic.config import ANALYSIS_WINDOW_SECONDS, IBI_BUFFER_RETENTION_MS
from biofizic.compute_features.hrv_metrics import compute_hrv_from_entries
from biofizic.compute_features.results import HrvMetrics, MultiWindowHrvResult
from biofizic.ingestion.messages import IbiBatchMessage, InterbeatIntervalEntry


class MultiWindowProcessor:
    """Runs the same HRV pipeline on 15, 30, 60, and 90 second lookbacks."""

    def __init__(
        self,
        window_seconds: tuple[int, ...] = ANALYSIS_WINDOW_SECONDS,
    ) -> None:
        self._window_seconds = window_seconds

    def compute(
        self,
        entries: list[InterbeatIntervalEntry],
        *,
        end_timestamp_ms: int | None = None,
    ) -> MultiWindowHrvResult:
        results: dict[int, HrvMetrics | None] = {}
        for seconds in self._window_seconds:
            span_ms = seconds * 1000
            if end_timestamp_ms is not None and end_timestamp_ms > 0:
                cutoff = end_timestamp_ms - span_ms
                window_entries = [
                    e
                    for e in entries
                    if e.timestamp_ms is None or e.timestamp_ms >= cutoff
                ]
            else:
                window_entries = entries
            results[seconds] = compute_hrv_from_entries(window_entries)

        return MultiWindowHrvResult(
            window_15_seconds=results.get(15),
            window_30_seconds=results.get(30),
            window_60_seconds=results.get(60),
            window_90_seconds=results.get(90),
        )


class RollingIbiBuffer:
    """Stores IBI entries with timestamp-based retention (default 120 s)."""

    def __init__(self, retention_ms: int = IBI_BUFFER_RETENTION_MS) -> None:
        self._retention_ms = retention_ms
        self._entries: deque[InterbeatIntervalEntry] = deque()

    def ingest_batch(self, batch: IbiBatchMessage) -> None:
        timestamps = batch.timestamps_ms or []
        if len(timestamps) == len(batch.intervals_ms):
            # Full per-beat timestamps from the watch: pair directly, dropping
            # only non-positive intervals.
            for ms, ts in zip(batch.intervals_ms, timestamps):
                if ms > 0:
                    self._entries.append(
                        InterbeatIntervalEntry(interval_ms=int(ms), timestamp_ms=int(ts))
                    )
        else:
            # Missing or partial timestamps (older zip() truncated and silently
            # dropped the tail beats). Reconstruct per-beat timestamps walking
            # backward from the batch anchor using the interval values, the same
            # way the watch does in buildIbiTimestamps. Never drop beats here.
            intervals = [int(ms) for ms in batch.intervals_ms if ms > 0]
            end_ts = int(batch.timestamp_ms)
            reconstructed = [0] * len(intervals)
            for i in range(len(intervals) - 1, -1, -1):
                reconstructed[i] = end_ts
                end_ts -= intervals[i]
            for ms, ts in zip(intervals, reconstructed):
                self._entries.append(
                    InterbeatIntervalEntry(interval_ms=ms, timestamp_ms=ts)
                )
        self._trim(batch.timestamp_ms)

    def _trim(self, now_ms: int) -> None:
        cutoff = now_ms - self._retention_ms
        while self._entries and (
            self._entries[0].timestamp_ms is not None
            and self._entries[0].timestamp_ms < cutoff
        ):
            self._entries.popleft()

    def entries_in_last_ms(self, span_ms: int, *, end_ms: int | None = None) -> list[InterbeatIntervalEntry]:
        if not self._entries:
            return []
        end = end_ms
        if end is None:
            timestamps = [e.timestamp_ms for e in self._entries if e.timestamp_ms]
            end = max(timestamps) if timestamps else 0
        if end <= 0:
            return list(self._entries)
        cutoff = end - span_ms
        return [
            e
            for e in self._entries
            if e.timestamp_ms is None or e.timestamp_ms >= cutoff
        ]


class RollingSensorBuffer:
    """Timestamped scalar samples (e.g. cardiac-band motion energy) with the
    same time-based retention as RollingIbiBuffer. The server only kept the
    latest SensorBatchMessage, so a windowed motion statistic had no history to
    draw on; this buffer provides the per-window median."""

    def __init__(self, retention_ms: int = IBI_BUFFER_RETENTION_MS) -> None:
        self._retention_ms = retention_ms
        self._samples: deque[tuple[int, float]] = deque()

    def ingest(self, timestamp_ms: int, value: float) -> None:
        self._samples.append((int(timestamp_ms), float(value)))
        cutoff = int(timestamp_ms) - self._retention_ms
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()

    def median_in_last_ms(self, span_ms: int, *, end_ms: int) -> float | None:
        if not self._samples or end_ms <= 0:
            return None
        cutoff = end_ms - span_ms
        values = sorted(v for ts, v in self._samples if ts >= cutoff)
        if not values:
            return None
        return values[len(values) // 2]
