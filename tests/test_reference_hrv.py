"""
Reference oracle (dev-only): our HRV math vs NeuroKit2 on the same IBI.

NeuroKit2 is a dev dependency, not shipped in production. The test skips when it
is not installed so the production suite stays light. When present, it proves
our RMSSD/SDNN agree with an independent, field-standard implementation.
"""

from __future__ import annotations

import numpy as np
import pytest

from biofizic.compute_features.hrv_metrics import compute_hrv_from_entries
from biofizic.ingestion.messages import InterbeatIntervalEntry

nk = pytest.importorskip("neurokit2", reason="NeuroKit2 not installed (dev-only oracle)")


def _entries_from_rri(rri_ms: list[int]) -> list[InterbeatIntervalEntry]:
    entries: list[InterbeatIntervalEntry] = []
    ts = 0
    for ibi in rri_ms:
        ts += ibi
        entries.append(InterbeatIntervalEntry(interval_ms=ibi, timestamp_ms=ts))
    return entries


def test_rmssd_sdnn_agree_with_neurokit():
    rng = np.random.default_rng(42)
    rri = [int(round(v)) for v in rng.normal(800, 30, size=120)]
    rri = [max(400, min(1400, v)) for v in rri]

    ours = compute_hrv_from_entries(_entries_from_rri(rri))
    assert ours is not None

    # NeuroKit2 time-domain HRV from R-peaks at 1000 Hz (1 sample = 1 ms).
    peaks = np.cumsum([0] + rri)
    hrv = nk.hrv_time(peaks, sampling_rate=1000, show=False)
    nk_rmssd = float(hrv["HRV_RMSSD"].iloc[0])
    nk_sdnn = float(hrv["HRV_SDNN"].iloc[0])

    assert ours.rmssd_ms == pytest.approx(nk_rmssd, rel=0.05)
    assert ours.sdnn_ms == pytest.approx(nk_sdnn, rel=0.05)
