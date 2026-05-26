"""
Golden synthetic signals: HRV math is checked against analytic ground truth.

For an IBI series alternating M+d, M-d, M+d, ... the successive differences are
+/-2d, so analytically:
    RMSSD = 2d,   SDNN = d,   mean HR = 60000 / M.
These give exact targets to validate the pipeline end of the math, independent
of any reference library.
"""

from __future__ import annotations

import pytest

from biofizic.compute_features.hrv_metrics import compute_hrv_from_entries
from biofizic.dsp.artifact_correction import correct_ibi_series
from biofizic.ingestion.messages import InterbeatIntervalEntry


def _alternating(mean_ms: int, d: int, n: int) -> list[InterbeatIntervalEntry]:
    entries: list[InterbeatIntervalEntry] = []
    ts = 0
    for i in range(n):
        ibi = mean_ms + d if i % 2 == 0 else mean_ms - d
        ts += ibi
        entries.append(InterbeatIntervalEntry(interval_ms=ibi, timestamp_ms=ts))
    return entries


def test_rmssd_sdnn_hr_match_analytic_truth():
    m = compute_hrv_from_entries(_alternating(mean_ms=800, d=20, n=40))
    assert m is not None
    assert m.rmssd_ms == pytest.approx(40.0, abs=1.0)   # 2d
    assert m.sdnn_ms == pytest.approx(20.0, abs=1.0)     # d
    assert m.mean_heart_rate_bpm == pytest.approx(75.0, abs=0.5)  # 60000/800
    assert m.artifact_rate == 0.0


def test_artifact_correction_recovers_clean_rmssd():
    clean = _alternating(mean_ms=800, d=20, n=40)
    clean_m = compute_hrv_from_entries(clean)

    dirty = list(clean)
    dirty[10] = InterbeatIntervalEntry(interval_ms=1700, timestamp_ms=dirty[10].timestamp_ms)
    dirty_m = compute_hrv_from_entries(dirty)

    assert dirty_m is not None and clean_m is not None
    # Corrected RMSSD stays close to the clean target; not blown up by the spike.
    assert dirty_m.rmssd_ms == pytest.approx(clean_m.rmssd_ms, abs=8.0)
    assert dirty_m.artifact_rate > 0.0


def test_correction_keeps_series_length():
    entries = _alternating(mean_ms=800, d=20, n=30)
    entries[5] = InterbeatIntervalEntry(interval_ms=50, timestamp_ms=entries[5].timestamp_ms)
    corrected, rate = correct_ibi_series(entries)
    assert len(corrected) == 30
    assert rate == pytest.approx(1 / 30)
