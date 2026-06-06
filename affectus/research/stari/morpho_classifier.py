"""Clasificator MORFOLOGIE PPG -> valenta (DISCONFORT vs PLACUT), axa 2 a cadranului.

Forma undei PPG (vasoconstrictie) separa disconfort de placut la 86% (WESAD 64Hz),
unde HRV cade la 52%. Consuma PPG brut 100Hz on-demand, il DOWNSAMPLE la 64Hz (paritate
EXACTA cu antrenarea WESAD), extrage features morfologice, prezice.

Research/viz, NU in verdictul de productie. Baseline personal acumulat live (repaus).
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
_TRAIN_FS = 64.0          # WESAD: modelul a vazut 64 Hz -> downsample 100->64 pt paritate
_WIN_S = 20
_BASELINE_MIN = 12
_BASELINE_WINDOW = 60
_MIN_SAMPLES = int(_WIN_S * 40)  # macar ~40 Hz worth ca sa merite


class MorphoClassifier:
    """Buffer PPG 100Hz on-demand -> downsample 64Hz -> morfologie -> 3 stari."""

    _BASELINE_FILE = _MODEL_PATH.parent / "morfologie_3_baseline.npz"

    def __init__(self) -> None:
        self._bundle = None
        self._ppg: deque[tuple[int, int]] = deque(maxlen=_WIN_S * 110)  # ts, green
        self._feat_buf: deque[np.ndarray] = deque(maxlen=_BASELINE_WINDOW)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
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
        """Primeste (ts_ms, green) la 100 Hz; tine ultimele ~20 s."""
        for ts, g in samples:
            self._ppg.append((ts, g))

    def _morph_features(self) -> np.ndarray | None:
        """Extrage forma undei din fereastra curenta, DOWNSAMPLED la 64 Hz.
        Features identice cu antrenarea: amp, rise_ms, width_ms, area (mean+std)."""
        if len(self._ppg) < _MIN_SAMPLES:
            return None
        ts = np.array([t for t, _ in self._ppg], dtype=float)
        g = np.array([v for _, v in self._ppg], dtype=float)
        dur_s = (ts[-1] - ts[0]) / 1000.0
        if dur_s < _WIN_S * 0.6:
            return None
        # downsample la 64 Hz: numarul de esantioane pt durata reala
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
