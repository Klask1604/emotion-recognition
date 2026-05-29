"""
Respiration comparator (research): RSA-from-IBI vs PPG-amplitude, side by side.

Runs both isolated estimators on each epoch and reports both rates, both
confidences, and their agreement, so the thesis can answer empirically: on
wrist PPG — and especially for a slow, quiet breather — which respiration source
is more reliable? Published only on biofizic/legacy/resp; never feeds the VR
decision. Neither estimator is fused into PhysiologyDecision until this
comparison justifies it (the B2d gate).
"""

from __future__ import annotations

from collections import deque

from biofizic.engine.channels.respiration_rsa import estimate_respiration_rsa
from biofizic.engine.channels.respiration_ppg import estimate_respiration_ppg
from biofizic.ingestion.messages import InterbeatIntervalEntry

# Respiration needs tens of seconds of signal, but the atomic batch carries only
# ~1 s. Accumulate a rolling window of IBI and raw PPG across batches.
_RESP_WINDOW_MS = 60_000


class RespirationCompareEngine:
    def __init__(self) -> None:
        # (timestamp_ms, interval_ms) and (timestamp_ms, green) rolling buffers.
        self._ibi: deque[tuple[int, int]] = deque()
        self._ppg: deque[tuple[int, int]] = deque()

    def _trim(self, buf: deque, now_ms: int) -> None:
        cutoff = now_ms - _RESP_WINDOW_MS
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def compute(self, batch) -> dict | None:
        """`batch` is the parsed AcquisitionBatchMessage. Accumulates IBI/PPG
        across batches (one batch is only ~1 s; respiration needs ~60 s) and
        runs both estimators on the rolling window. Returns a dict for
        biofizic/legacy/resp, or None when there is nothing to compute."""
        now_ms = int(batch.timestamp_anchor_ms or batch.timestamp_publish_ms or 0)

        # Append this batch's IBI to the rolling window.
        ibi_ms = batch.ibi_intervals_ms or []
        ibi_ts = batch.ibi_timestamps_ms or []
        if ibi_ms and len(ibi_ts) == len(ibi_ms):
            for m, t in zip(ibi_ms, ibi_ts):
                self._ibi.append((int(t), int(m)))
        elif ibi_ms:
            # No per-beat timestamps: approximate with the anchor (rare path).
            for m in ibi_ms:
                self._ibi.append((now_ms, int(m)))
        self._trim(self._ibi, now_ms)

        # Append this batch's raw PPG to the rolling window.
        if batch.ppg_green and batch.ppg_timestamps_ms and \
                len(batch.ppg_green) == len(batch.ppg_timestamps_ms):
            for t, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
                self._ppg.append((int(t), int(g)))
        self._trim(self._ppg, now_ms)

        # RSA over the accumulated IBI window.
        rsa = None
        if self._ibi:
            entries = [
                InterbeatIntervalEntry(interval_ms=m, timestamp_ms=t)
                for t, m in self._ibi
            ]
            rsa = estimate_respiration_rsa(entries)

        # PPG amplitude modulation over the accumulated PPG window.
        ppg = None
        if self._ppg:
            ts = [t for t, _ in self._ppg]
            green = [g for _, g in self._ppg]
            ppg = estimate_respiration_ppg(green, ts)

        if rsa is None and ppg is None:
            return None

        out: dict = {}
        if rsa is not None:
            out["rsa_bpm"] = round(rsa.breaths_per_min, 1)
            out["rsa_conf"] = round(rsa.confidence, 3)
            out["rsa_prom"] = round(rsa.prominence_ratio, 2)
        if ppg is not None:
            out["ppg_bpm"] = round(ppg.breaths_per_min, 1)
            out["ppg_conf"] = round(ppg.confidence, 3)
            out["ppg_prom"] = round(ppg.prominence_ratio, 2)

        # Agreement: absolute difference in br/min when BOTH are confident,
        # so the dashboard can show when the two sources actually corroborate.
        if (
            rsa is not None and ppg is not None
            and rsa.confidence > 0 and ppg.confidence > 0
        ):
            out["agree_bpm_diff"] = round(abs(rsa.breaths_per_min - ppg.breaths_per_min), 1)

        return out
