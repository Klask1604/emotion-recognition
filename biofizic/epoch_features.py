#!/usr/bin/env python3
"""
Backward-compatible re-exports for training scripts.
Canonical: biofizic.features.hrv_metrics
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biofizic.constants.hrv import (
    BAEVSKY_HISTOGRAM_BIN_MS,
    IBI_LOOKBACK_TRIM_MS,
    MAX_INTERBEAT_INTERVAL_MS,
    MAX_TIMESTAMP_IBI_MISMATCH_MS,
    MIN_BEATS_FOR_HRV,
    MIN_COVERED_SECONDS_FOR_HRV,
    MIN_INTERBEAT_INTERVAL_MS,
    OUTLIER_MEDIAN_DEVIATION_RATIO,
)
from biofizic.features.hrv_metrics import (
    compute_baevsky_indices,
    compute_hrv_from_entries,
    compute_hrv_from_mqtt_payload,
    warn_if_watch_server_rmssd_mismatch,
)
from biofizic.signal.ibi_cleaner import parse_intervals_from_payload, trim_entries_to_lookback
from biofizic.types.samples import HrvMetrics

FEATURE_NAMES = ["rmssd", "mean_hr", "sdnn", "mean_ibi", "pnn50"]

MIN_IBI_MS = MIN_INTERBEAT_INTERVAL_MS
MAX_IBI_MS = MAX_INTERBEAT_INTERVAL_MS
MEDIAN_DEV_RATIO = OUTLIER_MEDIAN_DEVIATION_RATIO
MAX_IBI_TS_MISMATCH_MS = MAX_TIMESTAMP_IBI_MISMATCH_MS
MIN_IBI_FOR_HRV = MIN_BEATS_FOR_HRV
MIN_WINDOW_SEC_FOR_SIGNAL = MIN_COVERED_SECONDS_FOR_HRV
EPOCH_IBI_TRIM_MS = IBI_LOOKBACK_TRIM_MS
BAEVSKY_BIN_MS = BAEVSKY_HISTOGRAM_BIN_MS


@dataclass(frozen=True)
class EpochFeatures:
    rmssd: float
    sdnn: float
    mean_ibi_ms: float
    mean_hr: float
    pnn50: float
    ibi_n: int
    window_sec: float
    baevsky_si: float = 0.0
    stress_index: float = 0.0

    @property
    def hrv_ready(self) -> bool:
        return self.ibi_n >= MIN_IBI_FOR_HRV and self.window_sec >= MIN_WINDOW_SEC_FOR_SIGNAL

    def as_vector(self) -> np.ndarray:
        return np.array(
            [self.rmssd, self.mean_hr, self.sdnn, self.mean_ibi_ms, self.pnn50],
            dtype=float,
        )


def _to_epoch(metrics: HrvMetrics | None) -> EpochFeatures | None:
    if metrics is None:
        return None
    return EpochFeatures(
        rmssd=metrics.rmssd_ms,
        sdnn=metrics.sdnn_ms,
        mean_ibi_ms=metrics.mean_interbeat_interval_ms,
        mean_hr=metrics.mean_heart_rate_bpm,
        pnn50=metrics.pnn50_percent,
        ibi_n=metrics.beat_count,
        window_sec=metrics.covered_seconds,
        baevsky_si=metrics.baevsky_stress_index_raw,
        stress_index=metrics.kubios_stress_index,
    )


def compute_from_ibi_entries(entries) -> EpochFeatures | None:
    from biofizic.types.samples import InterbeatIntervalEntry

    parsed: list[InterbeatIntervalEntry] = []
    for item in entries:
        if isinstance(item, InterbeatIntervalEntry):
            parsed.append(item)
        else:
            ms = int(item[0])
            ts = int(item[1]) if len(item) > 1 and item[1] is not None else None
            parsed.append(InterbeatIntervalEntry(interval_ms=ms, timestamp_ms=ts))
    return _to_epoch(compute_hrv_from_entries(parsed))


def compute_epoch_features(data: dict) -> EpochFeatures | None:
    return _to_epoch(compute_hrv_from_mqtt_payload(data))


def warn_if_watch_mismatch(data: dict, feats: EpochFeatures) -> None:
    m = HrvMetrics(
        rmssd_ms=feats.rmssd,
        sdnn_ms=feats.sdnn,
        mean_interbeat_interval_ms=feats.mean_ibi_ms,
        mean_heart_rate_bpm=feats.mean_hr,
        pnn50_percent=feats.pnn50,
        beat_count=feats.ibi_n,
        covered_seconds=feats.window_sec,
        baevsky_stress_index_raw=feats.baevsky_si,
        kubios_stress_index=feats.stress_index,
    )
    warn_if_watch_server_rmssd_mismatch(data, m)
