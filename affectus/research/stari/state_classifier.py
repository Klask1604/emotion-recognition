"""3-state classifier (CALM / DISCONFORT / PLACUT), the activation axis of the
Russell quadrant verdict.

Runs on the same 7 features as training (5 engine HRV + 2 motion: acc_rms,
acc_std) plus the personal deviation from a resting baseline accumulated over the
first calm windows. The personal baseline is the lever: it lifts the cross-subject
score from chance to ~65% LOSO (see personal_trainer for the per-user retrain).

Gyro is intentionally absent: the training set (WESAD E4) has no gyroscope, so the
model is acc-only. Gyro is still streamed and stored for future motion/gesture use,
just not fed to this verdict.

Model: train/results/validation/antreneaza_3stari_live.py -> models/stari_3.joblib.
The feature set MUST match the bundle's feature_names (7 abs + 7 deviation = 14
columns); train=serving parity is guaranteed by extracting the same 7 quantities
here as at training, with identical deviation [abs | (abs-mu)/sd].
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
# How many stable resting windows to accumulate for the personal baseline before
# classifying. Proven: a baseline from 6 windows drops -8pp; 12+ holds the score.
# ~2-3 min of rest.
_BASELINE_MIN = 12
_BASELINE_WINDOW = 60  # rolling, ca baseline-ul sa urmeze drift lent


class StateClassifier:
    """Incarca modelul o data; acumuleaza baseline personal pe 7 features; prezice."""

    _BASELINE_FILE = _MODEL_PATH.parent / "stari_3_baseline.npz"

    # Personal model retrained from the user's own labels. Preferred over the
    # generic one when present (see reload()). Lives in the mounted /data volume
    # so it survives a container rebuild (config picks /data vs local ./data).
    @staticmethod
    def _personal_path() -> Path:
        from affectus.config import personal_model_path
        return personal_model_path()

    def __init__(self) -> None:
        self._bundle = None
        self._is_personal = False  # True when the loaded bundle is the personal one
        self._feat_buf: deque[np.ndarray] = deque(maxlen=_BASELINE_WINDOW)
        self._mu: np.ndarray | None = None
        self._sd: np.ndarray | None = None
        self.last_features: list[float] | None = None  # last live vector, for feedback
        self.reload()
        # Persist baseline across restarts: don't re-calibrate ~3 min every time
        # the engine restarts. Loaded mu/sd let predict() work immediately.
        if self._BASELINE_FILE.exists():
            try:
                d = np.load(self._BASELINE_FILE)
                self._mu, self._sd = d["mu"], d["sd"]
            except Exception:
                pass

    def reload(self) -> bool:
        """(Re)load the model bundle, preferring the personal model if it exists.
        Safe to call after a retrain to swap it in WITHOUT an engine restart.
        Keeps the current bundle if loading fails. Returns True if personal."""
        if joblib is None:
            return self._is_personal
        for path, personal in ((self._personal_path(), True), (_MODEL_PATH, False)):
            if path.exists():
                try:
                    self._bundle = joblib.load(path)
                    self._is_personal = personal
                    return self._is_personal
                except Exception:
                    continue
        return self._is_personal

    @property
    def is_personal(self) -> bool:
        return self._is_personal

    def _save_baseline(self) -> None:
        try:
            np.savez(self._BASELINE_FILE, mu=self._mu, sd=self._sd)
        except Exception:
            pass

    @property
    def ready(self) -> bool:
        return self._bundle is not None

    def _features(self, window, sensor) -> np.ndarray | None:
        """The 7 features in the EXACT training order (models/stari_3.joblib):
        [rmssd, mean_hr, sdnn, mean_ibi, pnn50, acc_rms, acc_std].
        Read from the live WindowResult (result.best). mean_ibi = 60000/mean_hr.
        Train=serving parity: the same 7 quantities as at training, no approximations."""
        if window is None or float(getattr(window, "rmssd_ms", 0.0)) <= 0:
            return None
        mean_hr = float(window.mean_hr_bpm)
        if mean_hr <= 0:
            return None
        mean_ibi = 60000.0 / mean_hr  # mean IBI = 60000/HR, exact identity
        acc_rms = float(getattr(sensor, "acceleration_rms", 0.0) or 0.0) if sensor else 0.0
        acc_std = float(getattr(sensor, "acceleration_std", 0.0) or 0.0) if sensor else 0.0
        # Gyro is collected and stored, but NOT a classifier feature: the training
        # set (WESAD E4) has no gyroscope, so the model is acc-only. Feeding gyro
        # here would make 9 quantities (18 cols) against a 7-feature (14-col) model
        # and crash predict_proba. Gyro stays for future motion/gesture use, not
        # this verdict. See state_classifier docstring + gyroscope-role note.
        return np.array([
            float(window.rmssd_ms),
            mean_hr,
            float(window.sdnn_ms),
            mean_ibi,
            float(window.pnn50_pct),
            acc_rms, acc_std,
        ], dtype=float)

    def observe_rest(self, primary, sensor) -> None:
        """Feed the personal baseline with one stable resting window (motion still)."""
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
        """Return {state, confidence, probs}, or None if not ready. Requires the
        accumulated personal baseline (same as at training)."""
        if not self.ready or self._mu is None:
            return None
        f = self._features(primary, sensor)
        if f is None:
            return None
        self.last_features = f.tolist()  # cached so a feedback tap can store it
        dev = (f - self._mu) / self._sd
        x = np.hstack([f, dev]).reshape(1, -1)  # [abs | deviation], as at training
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
