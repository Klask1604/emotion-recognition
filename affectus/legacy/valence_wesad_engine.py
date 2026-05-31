"""
WESAD valence research engine (parallel, never feeds the VR decision).

Accumulates raw PPG across batches and, every epoch, runs the WESAD-trained
valence model (stress vs amusement = the valence axis, ~85% LOSO) over the
rolling window, publishing its live prediction on biofizic/legacy/valence_wesad
for the comparison dashboard. This is the SAME model and the SAME unified
extractor used by the production fusion channel (engine/channels/valence_wesad),
but here it is observed-only: we log its prediction next to the production
verdict to judge how well it transfers to the watch BEFORE trusting it in the
decision.

Mirrors ValenceFdEngine's rolling-buffer pattern; mirrors legacy WesadEngine's
graceful FileNotFoundError handling (toggle on but model not trained -> skip).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from affectus.engine.channels.valence_wesad import ValenceWesadState
from affectus.legacy.valence_features import extract_valence_feature_vector

_PPG_WINDOW_MS = 20_000  # same epoch as the FD engine / the training window


class ValenceWesadEngine:
    """Rolling-PPG WESAD valence predictor. Raises FileNotFoundError when the
    model file is absent, so the facade can skip it gracefully like WesadEngine."""

    def __init__(self, model_path: Path | None = None) -> None:
        self._state = ValenceWesadState()
        self._state.ensure_loaded(model_path)
        if self._state.model is None:
            raise FileNotFoundError(
                "models/valence_wesad.joblib not found — train it with "
                "train/train_valence_wesad.py"
            )
        self._ppg: deque[tuple[int, int]] = deque()      # 25 Hz from batch
        self._ppg_hi: deque[tuple[int, int]] = deque()   # 100 Hz on-demand

    def _trim(self, buf: deque, now_ms: int) -> None:
        cutoff = now_ms - _PPG_WINDOW_MS
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def ingest_ondemand_ppg(self, samples: list[tuple[int, int]]) -> None:
        """Feed 100 Hz on-demand PPG (list of (ts_ms, green)) for sharper
        vascular/morphology features when available."""
        for t, g in samples:
            self._ppg_hi.append((int(t), int(g)))
        if self._ppg_hi:
            self._trim(self._ppg_hi, self._ppg_hi[-1][0])

    def compute(self, batch) -> dict | None:
        now_ms = int(batch.timestamp_anchor_ms or batch.timestamp_publish_ms or 0)
        if batch.ppg_green and batch.ppg_timestamps_ms and \
                len(batch.ppg_green) == len(batch.ppg_timestamps_ms):
            for t, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
                self._ppg.append((int(t), int(g)))
        self._trim(self._ppg, now_ms)
        self._trim(self._ppg_hi, now_ms)

        # Prefer the 100 Hz on-demand buffer when it has a usable window.
        hi_span = (self._ppg_hi[-1][0] - self._ppg_hi[0][0]) if len(self._ppg_hi) > 1 else 0
        use_hi = len(self._ppg_hi) >= 256 and hi_span >= _PPG_WINDOW_MS * 0.5
        buf = self._ppg_hi if use_hi else self._ppg
        if len(buf) < 64:
            return None

        ts = [t for t, _ in buf]
        green = [g for _, g in buf]
        vec = extract_valence_feature_vector(green, ts)
        if vec is None:
            return None
        x = (np.asarray(vec, dtype=float) - self._state.feature_mean) / self._state.feature_std
        try:
            proba = self._state.model.predict_proba(x.reshape(1, -1))[0]
        except Exception:  # noqa: BLE001
            return None
        p_pos = float(proba[1])          # class 1 = positive valence (amusement)
        z = 2.0 * p_pos - 1.0            # [0,1] -> [-1, 1], 0 = neutral
        return {
            "p_positive": round(p_pos, 4),
            "valence_z": round(z, 4),     # >0 positive, <0 negative
            "confidence": round(abs(z), 4),
            "src_hz": round((len(buf) - 1) / ((ts[-1] - ts[0]) / 1000.0), 0)
                       if ts[-1] > ts[0] else 0,
        }
