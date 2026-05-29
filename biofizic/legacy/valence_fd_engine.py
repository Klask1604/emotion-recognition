"""
Valence frequency-domain engine: accumulates raw PPG across batches and emits
the nine PPG-FD features (see valence_ppg_fd) on biofizic/legacy/valence_fd.

Like the respiration comparator, one acquisition batch carries only ~1 s of PPG,
but the FFT needs a ~20 s window (the paper's epoch), so we keep a rolling PPG
buffer. The HR fundamental is taken from the batch's reported HR.

This engine extracts FEATURES only — it does not output a valence verdict,
because a trustworthy verdict needs an SVM trained on labelled data we do not
have for the watch. The features feed (a) a thesis demonstration that the SOTA
method is computable on consumer PPG, and (b) future classifier training against
ground-truth. It NEVER feeds the production decision.
"""

from __future__ import annotations

from collections import deque

from biofizic.legacy.valence_ppg_fd import extract_valence_fd_features

_PPG_WINDOW_MS = 20_000  # the paper's 20 s epoch


class ValenceFdEngine:
    def __init__(self) -> None:
        self._ppg: deque[tuple[int, int]] = deque()  # (ts_ms, green)
        self._last_hr: float = 0.0

    def compute(self, batch) -> dict | None:
        now_ms = int(batch.timestamp_anchor_ms or batch.timestamp_publish_ms or 0)
        if batch.heart_rate_bpm and batch.heart_rate_bpm > 0:
            self._last_hr = float(batch.heart_rate_bpm)

        if batch.ppg_green and batch.ppg_timestamps_ms and \
                len(batch.ppg_green) == len(batch.ppg_timestamps_ms):
            for t, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
                self._ppg.append((int(t), int(g)))
        # Trim to the rolling window.
        cutoff = now_ms - _PPG_WINDOW_MS
        while self._ppg and self._ppg[0][0] < cutoff:
            self._ppg.popleft()

        if not self._ppg or self._last_hr <= 0:
            return None

        ts = [t for t, _ in self._ppg]
        green = [g for _, g in self._ppg]
        feats = extract_valence_fd_features(green, ts, hr_bpm=self._last_hr)
        if not feats.valid:
            return None
        return feats.as_dict()
