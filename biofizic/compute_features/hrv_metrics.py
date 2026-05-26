"""
Compute RMSSD, SDNN, pNN50, Baevsky SI, and Kubios stress index from IBI.

Formulas:
  RMSSD = sqrt(mean((IBI[i+1] - IBI[i])^2)) in milliseconds
  SDNN  = standard deviation of IBI values
  pNN50 = percentage of successive differences > 50 ms
  Baevsky SI = AMo / (2 * Mo * MxDMn)
    Mo    = median RR interval in seconds
    AMo   = percent of intervals in the 50 ms bin containing Mo
    MxDMn = max(RR) - min(RR) in seconds
  Kubios stress_index = sqrt(Baevsky SI)
"""

from __future__ import annotations

import numpy as np

from biofizic.config import BAEVSKY_HISTOGRAM_BIN_MS
from biofizic.dsp.artifact_correction import correct_ibi_series
from biofizic.dsp.ibi_filter import successive_interval_differences
from biofizic.compute_features.results import HrvMetrics
from biofizic.ingestion.messages import InterbeatIntervalEntry

FEATURE_VECTOR_NAMES = ["rmssd", "mean_hr", "sdnn", "mean_ibi", "pnn50"]


def compute_baevsky_indices(intervals_ms: np.ndarray) -> tuple[float, float]:
    if intervals_ms.size < 4:
        return 0.0, 0.0

    rr_seconds = intervals_ms.astype(float) / 1000.0
    mode_rr = float(np.median(rr_seconds))
    if mode_rr <= 0:
        return 0.0, 0.0

    range_rr = float(np.max(rr_seconds) - np.min(rr_seconds))
    if range_rr <= 1e-6:
        return 0.0, 0.0

    bin_half = BAEVSKY_HISTOGRAM_BIN_MS / 2000.0
    in_mode_bin = np.sum(np.abs(rr_seconds - mode_rr) <= bin_half)
    amplitude_mode_percent = 100.0 * in_mode_bin / rr_seconds.size

    raw_index = amplitude_mode_percent / (2.0 * mode_rr * range_rr)
    if not np.isfinite(raw_index) or raw_index <= 0:
        return 0.0, 0.0
    return float(raw_index), float(np.sqrt(raw_index))


def compute_hrv_from_entries(
    entries: list[InterbeatIntervalEntry],
) -> HrvMetrics | None:
    # Correct artifacts by interpolation (not deletion) so RMSSD/SI are computed
    # on a continuous series; artifact_rate is the fraction that was corrected.
    corrected, artifact_rate = correct_ibi_series(entries)
    if len(corrected) < 2:
        return None
    valid = corrected

    values = np.array([e.interval_ms for e in valid], dtype=float)
    mean_interval = float(np.mean(values))
    sdnn = float(np.std(values, ddof=0))

    diffs = successive_interval_differences(valid)
    if diffs:
        diffs_arr = np.array(diffs, dtype=float)
        rmssd = float(np.sqrt(np.mean(diffs_arr**2)))
        pnn50 = float(100.0 * np.sum(np.abs(diffs_arr) > 50) / len(diffs_arr))
    else:
        rmssd = 0.0
        pnn50 = 0.0

    mean_hr = 60_000.0 / mean_interval if mean_interval > 0 else 0.0
    covered_seconds = float(np.sum(values) / 1000.0)
    baevsky_raw, kubios_stress = compute_baevsky_indices(values)

    return HrvMetrics(
        rmssd_ms=rmssd,
        sdnn_ms=sdnn,
        mean_interbeat_interval_ms=mean_interval,
        mean_heart_rate_bpm=mean_hr,
        pnn50_percent=pnn50,
        beat_count=len(valid),
        covered_seconds=covered_seconds,
        baevsky_stress_index_raw=baevsky_raw,
        kubios_stress_index=kubios_stress,
        artifact_rate=artifact_rate,
    )
