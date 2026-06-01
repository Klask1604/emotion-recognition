"""
Polarity research engine (parallel, never feeds the VR decision).

Answers the product question directly: is the affect NEGATIVE or POSITIVE right
now? Trained on WESAD wrist BVP stress-vs-amusement (both high-arousal, so the
model is forced onto the affect-type signal, ~81% LOSO) with a CORAL transform
baked into the bundle that aligns each feature window to the watch's own PPG
distribution before predicting - the domain-shift fix that makes it trustworthy
on the real signal.

Mirrors ValenceWesadEngine's rolling-buffer pattern. Observed-only on
biofizic/legacy/polarity; never touches the production PhysiologyDecision.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

from affectus.legacy.valence_features import extract_valence_feature_vector

_PPG_WINDOW_MS = 20_000


class PolarityEngine:
    """Rolling-PPG negative-vs-positive predictor. Raises FileNotFoundError when
    the bundle is absent, so the facade can skip it gracefully."""

    def __init__(self, model_path: Path | None = None) -> None:
        path = model_path or (Path(__file__).resolve().parents[2] / "models" / "polarity.joblib")
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - train it with train/models/train_polarity.py"
            )
        import joblib
        b = joblib.load(path)
        self._model = b["model"]
        self._feature_mean = np.asarray(b["feature_mean"], float)
        self._feature_std = np.asarray(b["feature_std"], float)
        # CORAL transform pieces (applied before normalisation)
        self._coral_A = np.asarray(b["coral_A"], float)
        self._coral_mu_s = np.asarray(b["coral_mu_s"], float)
        self._coral_sd_s = np.asarray(b["coral_sd_s"], float)
        self._coral_sd_t = np.asarray(b["coral_sd_t"], float)
        self._ppg: deque[tuple[int, int]] = deque()      # 25 Hz from batch
        self._ppg_hi: deque[tuple[int, int]] = deque()   # 100 Hz on-demand

    def _trim(self, buf: deque, now_ms: int) -> None:
        cutoff = now_ms - _PPG_WINDOW_MS
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def ingest_ondemand_ppg(self, samples: list[tuple[int, int]]) -> None:
        for t, g in samples:
            self._ppg_hi.append((int(t), int(g)))
        if self._ppg_hi:
            self._trim(self._ppg_hi, self._ppg_hi[-1][0])

    def _apply_coral(self, vec: np.ndarray) -> np.ndarray:
        """Map a raw feature window into the watch's aligned distribution."""
        z = (vec - self._coral_mu_s) / self._coral_sd_s
        return z @ self._coral_A * self._coral_sd_t + self._coral_mu_s

    def compute(self, batch) -> dict | None:
        now_ms = int(batch.timestamp_anchor_ms or batch.timestamp_publish_ms or 0)
        if batch.ppg_green and batch.ppg_timestamps_ms and \
                len(batch.ppg_green) == len(batch.ppg_timestamps_ms):
            for t, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
                self._ppg.append((int(t), int(g)))
        self._trim(self._ppg, now_ms)
        self._trim(self._ppg_hi, now_ms)

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

        aligned = self._apply_coral(np.asarray(vec, float))
        x = (aligned - self._feature_mean) / self._feature_std
        try:
            proba = self._model.predict_proba(x.reshape(1, -1))[0]
        except Exception:  # noqa: BLE001
            return None
        # 3-class model: index 0=neutral, 1=negative, 2=positive (bundle classes).
        p_neutral, p_negative, p_positive = (float(proba[0]), float(proba[1]),
                                             float(proba[2]))
        idx = int(np.argmax(proba))
        label = ["neutral", "negative", "positive"][idx]
        label_code = {"neutral": 0, "negative": -1, "positive": 1}[label]
        # signed score for the timeseries: positive prob minus negative prob, so
        # >0 leans positive, <0 leans negative, ~0 = neutral.
        z = p_positive - p_negative
        return {
            "p_neutral": round(p_neutral, 4),
            "p_negative": round(p_negative, 4),
            "p_positive": round(p_positive, 4),
            "polarity_z": round(z, 4),         # >0 positive, <0 negative
            "confidence": round(float(proba[idx]), 4),   # prob of the chosen class
            "label": label,
            "label_code": label_code,
        }
