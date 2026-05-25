"""Compute HRV metrics for multiple window lengths in parallel."""

from __future__ import annotations

from biofizic.constants.hrv import ANALYSIS_WINDOW_SECONDS
from biofizic.features.hrv_metrics import compute_hrv_from_entries
from biofizic.types.samples import HrvMetrics, InterbeatIntervalEntry, MultiWindowHrvResult


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
