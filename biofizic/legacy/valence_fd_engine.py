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
    """Accumulates raw PPG (25 Hz from the batch OR 100 Hz on-demand) and emits
    the nine FD features. On-demand 100 Hz, when present, supersedes the 25 Hz
    batch PPG for higher spectral resolution; otherwise the batch feeds it."""

    def __init__(self) -> None:
        self._ppg: deque[tuple[int, int]] = deque()      # 25 Hz from batch
        self._ppg_hi: deque[tuple[int, int]] = deque()   # 100 Hz on-demand
        self._last_hr: float = 0.0

    def _trim(self, buf: deque, now_ms: int) -> None:
        cutoff = now_ms - _PPG_WINDOW_MS
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def ingest_ondemand_ppg(self, samples: list[tuple[int, int]]) -> None:
        """Feed 100 Hz on-demand PPG: list of (ts_ms, green)."""
        for t, g in samples:
            self._ppg_hi.append((int(t), int(g)))
        if self._ppg_hi:
            self._trim(self._ppg_hi, self._ppg_hi[-1][0])

    def compute(self, batch) -> dict | None:
        now_ms = int(batch.timestamp_anchor_ms or batch.timestamp_publish_ms or 0)
        if batch.heart_rate_bpm and batch.heart_rate_bpm > 0:
            self._last_hr = float(batch.heart_rate_bpm)

        if batch.ppg_green and batch.ppg_timestamps_ms and \
                len(batch.ppg_green) == len(batch.ppg_timestamps_ms):
            for t, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
                self._ppg.append((int(t), int(g)))
        self._trim(self._ppg, now_ms)
        self._trim(self._ppg_hi, now_ms)

        if self._last_hr <= 0:
            return None

        # Prefer the 100 Hz on-demand buffer when it has a full window.
        hi_span = (self._ppg_hi[-1][0] - self._ppg_hi[0][0]) if len(self._ppg_hi) > 1 else 0
        use_hi = len(self._ppg_hi) >= 256 and hi_span >= _PPG_WINDOW_MS * 0.5
        buf = self._ppg_hi if use_hi else self._ppg
        if not buf:
            return None

        ts = [t for t, _ in buf]
        green = [g for _, g in buf]
        feats = extract_valence_fd_features(green, ts, hr_bpm=self._last_hr)
        if not feats.valid:
            return None
        out = feats.as_dict()
        out["src_hz"] = round((len(buf) - 1) / ((ts[-1] - ts[0]) / 1000.0), 0) if ts[-1] > ts[0] else 0
        return out
