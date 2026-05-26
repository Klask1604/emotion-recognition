"""Compute HRV metrics for multiple window lengths and rolling buffers."""

from __future__ import annotations

from collections import deque

from biofizic.config import ANALYSIS_WINDOW_SECONDS, IBI_BUFFER_RETENTION_MS
from biofizic.features.hrv_metrics import compute_hrv_from_entries
from biofizic.types import (
    HrvMetrics,
    IbiBatchMessage,
    InterbeatIntervalEntry,
    MultiWindowHrvResult,
)


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
        for ms, ts in zip(batch.intervals_ms, batch.timestamps_ms or []):
            if ms > 0:
                self._entries.append(
                    InterbeatIntervalEntry(interval_ms=int(ms), timestamp_ms=int(ts))
                )
        if not batch.timestamps_ms:
            for ms in batch.intervals_ms:
                if ms > 0:
                    self._entries.append(
                        InterbeatIntervalEntry(
                            interval_ms=int(ms), timestamp_ms=batch.timestamp_ms
                        )
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
