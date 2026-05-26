"""
Artifact correction for IBI series (interpolation, not deletion).

HRV practice (Task Force ESC/NASPE 1996; Kubios HRV artifact correction) is to
*correct* ectopic / artifact beats by interpolation rather than to delete them:
deleting beats shortens the series, biases the sample and destabilises RMSSD
(which depends on successive differences). We flag beats that are either non-
physiological (outside [MIN, MAX] ms) or deviate from the local median by more
than the outlier ratio, then replace the flagged interval values by linear
interpolation over the surviving beats, keeping every beat's position and
timestamp. The fraction of corrected beats is returned as the artifact rate,
a standard signal-quality index.
"""

from __future__ import annotations

import numpy as np

from biofizic.config import (
    LOCAL_MEDIAN_HALF_WINDOW,
    MAX_INTERBEAT_INTERVAL_MS,
    MIN_INTERBEAT_INTERVAL_MS,
    OUTLIER_MEDIAN_DEVIATION_RATIO,
)
from biofizic.ingestion.messages import InterbeatIntervalEntry


def correct_ibi_series(
    entries: list[InterbeatIntervalEntry],
) -> tuple[list[InterbeatIntervalEntry], float]:
    """Return (corrected entries, artifact_rate).

    A beat is an artifact when it is out of physiological range OR deviates from
    the median of its LOCAL neighbours by more than OUTLIER_MEDIAN_DEVIATION_RATIO.
    Using the local (not whole-window) median is what prevents genuine HRV from
    being mislabelled as artifacts. Flagged beats are replaced by linear
    interpolation over the valid beats; every beat keeps its position/timestamp.
    If fewer than 2 valid beats survive, returns ([], artifact_rate).
    """
    n = len(entries)
    if n == 0:
        return [], 0.0

    intervals = np.array([e.interval_ms for e in entries], dtype=float)
    valid = (intervals >= MIN_INTERBEAT_INTERVAL_MS) & (intervals <= MAX_INTERBEAT_INTERVAL_MS)

    if valid.sum() >= 2:
        w = LOCAL_MEDIAN_HALF_WINDOW
        range_valid = valid.copy()
        for i in range(n):
            if not range_valid[i]:
                continue
            lo = max(0, i - w)
            hi = min(n, i + w + 1)
            neigh = intervals[lo:hi][range_valid[lo:hi]]
            # The median of ~2w+1 beats is robust to a single spike, so a beat
            # that is itself an artifact still sees a clean local median.
            if neigh.size < 2:
                continue
            local_median = float(np.median(neigh))
            if abs(intervals[i] - local_median) >= OUTLIER_MEDIAN_DEVIATION_RATIO * local_median:
                valid[i] = False

    artifact_rate = 1.0 - float(valid.sum()) / n
    if valid.sum() < 2:
        return [], artifact_rate

    corrected = intervals.copy()
    if (~valid).any():
        idx = np.arange(n)
        corrected[~valid] = np.interp(idx[~valid], idx[valid], intervals[valid])

    out = [
        InterbeatIntervalEntry(
            interval_ms=int(round(corrected[i])),
            timestamp_ms=entries[i].timestamp_ms,
        )
        for i in range(n)
    ]
    return out, artifact_rate
