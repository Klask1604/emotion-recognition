"""PPG morphology classifier -> valence (DISCONFORT vs PLACUT), the 2nd quadrant axis.

The pulse-wave shape (vasoconstriction) separates discomfort from pleasant at 86%
(WESAD 64 Hz), where HRV alone collapses to 52%. It consumes the raw 100 Hz PPG,
DOWNSAMPLES it to 64 Hz (exact parity with WESAD training), extracts fs-invariant
morphology features (rise time, width, amplitude, area), and predicts. Valence is
only refined in the activated half of the plane (where the shape carries it).

Personal baseline accumulated live at rest.
Model: train/results/validation/antreneaza_morfologie.py -> models/morfologie_3.joblib
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, resample

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "morfologie_3.joblib"
_TRAIN_FS = 64.0          # WESAD: the model saw 64 Hz -> downsample 100->64 for parity
_WIN_S = 20
_BASELINE_MIN = 12
_BASELINE_WINDOW = 60
_MIN_SAMPLES = int(_WIN_S * 40)  # at least ~40 Hz worth to be usable


class MorphoClassifier:
    """Buffer 100 Hz on-demand PPG -> downsample to 64 Hz -> morphology -> 3 states."""

    _BASELINE_FILE = _MODEL_PATH.parent / "morfologie_3_baseline.npz"

    def __init__(self) -> None:
        self._bundle = None
        self._ppg: deque[tuple[int, int]] = deque(maxlen=_WIN_S * 110)  # ts, green
        self._feat_buf: deque[np.ndarray] = deque(maxlen=_BASELINE_WINDOW)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self.last_features: list[float] | None = None  # last live vector, for feedback
        if joblib is not None and _MODEL_PATH.exists():
            try:
                self._bundle = joblib.load(_MODEL_PATH)
            except Exception:
                self._bundle = None
        if self._BASELINE_FILE.exists():
            try:
                d = np.load(self._BASELINE_FILE)
                self._mu, self._sd = d["mu"], d["sd"]
            except Exception:
                pass

    def _save_baseline(self) -> None:
        try:
            np.savez(self._BASELINE_FILE, mu=self._mu, sd=self._sd)
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    def ingest_ondemand_ppg(self, samples: list[tuple[int, int]]) -> None:
        """Receive (ts_ms, green) at 100 Hz; keep the last ~20 s."""
        for ts, g in samples:
            self._ppg.append((ts, g))

    def _morph_features(self) -> np.ndarray | None:
        """Extract the pulse-wave shape from the current window, DOWNSAMPLED to 64 Hz.
        Features identical to training: amp, rise_ms, width_ms, area (mean+std)."""
        if len(self._ppg) < _MIN_SAMPLES:
            return None
        ts = np.array([t for t, _ in self._ppg], dtype=float)
        g = np.array([v for _, v in self._ppg], dtype=float)
        dur_s = (ts[-1] - ts[0]) / 1000.0
        if dur_s < _WIN_S * 0.6:
            return None
        # downsample to 64 Hz: sample count for the real duration
        n64 = max(8, int(dur_s * _TRAIN_FS))
        sig = resample(g, n64)
        fs = _TRAIN_FS
        s = sig - np.mean(sig)
        if np.std(s) < 1e-6:
            return None
        b, a = butter(2, [0.5 / (fs / 2), min(8, fs / 2 - 0.1) / (fs / 2)], btype="band")
        f = filtfilt(b, a, s)
        pk, _ = find_peaks(f, distance=int(0.4 * fs))
        if len(pk) < 4:
            return None
        amps, rises, widths, areas = [], [], [], []
        for i in range(1, len(pk)):
            seg = f[pk[i - 1]:pk[i]]
            if len(seg) < 3:
                continue
            amp = f[pk[i]] - seg.min()
            amps.append(amp)
            mn = int(np.argmin(seg))
            rises.append((len(seg) - mn) / fs * 1000)
            half = seg.min() + amp * 0.5
            widths.append(np.sum(seg > half) / fs * 1000)
            areas.append(np.trapezoid(seg - seg.min()) / fs)
        if len(amps) < 3:
            return None
        def st(x): return [float(np.mean(x)), float(np.std(x))]
        return np.array(st(amps) + st(rises) + st(widths) + st(areas), dtype=float)

    def observe_rest(self) -> None:
        f = self._morph_features()
        if f is None:
            return
        self._feat_buf.append(f)
        if len(self._feat_buf) >= _BASELINE_MIN:
            arr = np.array(self._feat_buf)
            self._mu = arr.mean(0)
            self._sd = arr.std(0) + 1e-8
            self._save_baseline()

    def predict(self) -> dict | None:
        if not self.ready or self._mu is None:
            return None
        f = self._morph_features()
        if f is None:
            return None
        self.last_features = f.tolist()  # cached so a feedback tap can store it
        dev = (f - self._mu) / self._sd
        x = np.hstack([f, dev]).reshape(1, -1)
        model = self._bundle["model"]
        classes = self._bundle["classes"]
        probs = model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return {
            "state": classes[idx],
            "confidence": round(float(probs[idx]), 3),
            "probs": {classes[i]: round(float(probs[i]), 3) for i in range(len(probs))},
        }
