"""Clasificator de STARE in 3 niveluri (NEUTRU/STRES/RELAXARE) din modelul WESAD.

Research / vizualizare — NU intra in verdictul de productie (arousal). Ruleaza pe
aceleasi 7 features ca antrenarea (5 HRV engine + 2 motion) + deviatie personala
fata de un baseline propriu acumulat in primele ferestre de repaus.

Modelul: train/results/validation/antreneaza_3stari_live.py -> models/stari_3.joblib
Paritate train=serving garantata: aceleasi 7 features, aceeasi deviatie [abs | (abs-mu)/sd].
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np

try:
    import joblib
except Exception:  # pragma: no cover
    joblib = None

_MODEL_PATH = Path(__file__).resolve().parents[3] / "models" / "stari_3.joblib"
# Cate ferestre de repaus stabil acumulam pt baseline personal inainte de a clasifica.
# Dovedit (plan_rapid.py): baseline din 6 ferestre CADE -8pp; 12+ tine cifra. ~2-3 min repaus.
_BASELINE_MIN = 12
_BASELINE_WINDOW = 60  # rolling, ca baseline-ul sa urmeze drift lent


class StateClassifier:
    """Incarca modelul o data; acumuleaza baseline personal pe 7 features; prezice."""

    _BASELINE_FILE = _MODEL_PATH.parent / "stari_3_baseline.npz"

    def __init__(self) -> None:
        self._bundle = None
        self._feat_buf: deque[np.ndarray] = deque(maxlen=_BASELINE_WINDOW)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        if joblib is not None and _MODEL_PATH.exists():
            try:
                self._bundle = joblib.load(_MODEL_PATH)
            except Exception:
                self._bundle = None
        # Persist baseline across restarts: don't re-calibrate ~3 min every time
        # the engine restarts. Loaded mu/sd let predict() work immediately.
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

    def _features(self, window, sensor) -> np.ndarray | None:
        """Cele 9 features in ORDINEA EXACTA a antrenarii (models/stari_3.joblib):
        [rmssd, mean_hr, sdnn, mean_ibi, pnn50, acc_rms, acc_std, gyro_rms, gyro_std].
        Citeste din WindowResult-ul live (result.best). mean_ibi = 60000/mean_hr (exact).
        Paritate train=serving: aceleasi 9 marimi ca la antrenare, zero aproximari."""
        if window is None or float(getattr(window, "rmssd_ms", 0.0)) <= 0:
            return None
        mean_hr = float(window.mean_hr_bpm)
        if mean_hr <= 0:
            return None
        mean_ibi = 60000.0 / mean_hr  # IBI mediu = 60000/HR, identitate exacta
        acc_rms = float(getattr(sensor, "acceleration_rms", 0.0) or 0.0) if sensor else 0.0
        acc_std = float(getattr(sensor, "acceleration_std", 0.0) or 0.0) if sensor else 0.0
        gyro_rms = float(getattr(sensor, "gyroscope_rms", 0.0) or 0.0) if sensor else 0.0
        gyro_std = float(getattr(sensor, "gyroscope_std", 0.0) or 0.0) if sensor else 0.0
        return np.array([
            float(window.rmssd_ms),
            mean_hr,
            float(window.sdnn_ms),
            mean_ibi,
            float(window.pnn50_pct),
            acc_rms, acc_std, gyro_rms, gyro_std,
        ], dtype=float)

    def observe_rest(self, primary, sensor) -> None:
        """Hraneste baseline-ul personal cu o fereastra de repaus stabil (motion still)."""
        f = self._features(primary, sensor)
        if f is None:
            return
        self._feat_buf.append(f)
        if len(self._feat_buf) >= _BASELINE_MIN:
            arr = np.array(self._feat_buf)
            self._mu = arr.mean(0)
            self._sd = arr.std(0) + 1e-8
            self._save_baseline()

    def predict(self, primary, sensor) -> dict | None:
        """Returneaza {state, confidence, probs} sau None daca nu e gata.
        Necesita baseline personal acumulat (la fel ca antrenarea)."""
        if not self.ready or self._mu is None:
            return None
        f = self._features(primary, sensor)
        if f is None:
            return None
        dev = (f - self._mu) / self._sd
        x = np.hstack([f, dev]).reshape(1, -1)  # [abs | deviatie] = ca la antrenare
        model = self._bundle["model"]
        classes = self._bundle["classes"]  # {0:NEUTRU,1:STRES,2:RELAXARE}
        probs = model.predict_proba(x)[0]
        idx = int(np.argmax(probs))
        return {
            "state": classes[idx],
            "confidence": round(float(probs[idx]), 3),
            "probs": {classes[i]: round(float(probs[i]), 3) for i in range(len(probs))},
            "baseline_ready": True,
        }
