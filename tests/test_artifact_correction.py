"""IBI artifact correction: interpolate (not delete) and report artifact rate."""

from __future__ import annotations

import pytest

from biofizic.dsp.artifact_correction import correct_ibi_series
from biofizic.ingestion.messages import InterbeatIntervalEntry


def _series(values: list[int]) -> list[InterbeatIntervalEntry]:
    return [InterbeatIntervalEntry(interval_ms=v, timestamp_ms=i * 800) for i, v in enumerate(values)]


def test_clean_series_is_unchanged_zero_artifacts():
    corrected, rate = correct_ibi_series(_series([800, 810, 790, 805, 800]))
    assert rate == 0.0
    assert [e.interval_ms for e in corrected] == [800, 810, 790, 805, 800]


def test_outlier_is_interpolated_not_dropped():
    # One spike at index 2; correction must keep length 5 and replace the spike
    # with a value near its neighbours (not delete it).
    corrected, rate = correct_ibi_series(_series([800, 810, 1500, 805, 800]))
    assert len(corrected) == 5  # not shortened
    assert rate == pytest.approx(0.2)
    assert 780 <= corrected[2].interval_ms <= 815  # interpolated toward neighbours


def test_correction_stabilises_rmssd_vs_deletion():
    # A series with one artifact: corrected RMSSD should be close to the clean
    # RMSSD, far smaller than if the artifact were kept.
    import numpy as np

    clean = [800, 805, 798, 802, 800, 804, 799]
    dirty = list(clean)
    dirty[3] = 1600  # injected artifact
    corrected, _ = correct_ibi_series(_series(dirty))
    vals = np.array([e.interval_ms for e in corrected], dtype=float)
    rmssd = float(np.sqrt(np.mean(np.diff(vals) ** 2)))
    assert rmssd < 50  # not blown up by the 1600 ms spike
