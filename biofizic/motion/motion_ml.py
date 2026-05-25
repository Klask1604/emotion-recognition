"""WISDM Random Forest HAR inference on wrist motion features."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np

from biofizic.motion.motion_calibrator import MotionCalibrator
from biofizic.motion.motion_features import MotionFeatureVector

MOTION_CLASSES = ("STILL", "SCROLL", "HAND", "WALK")


@dataclass(frozen=True)
class MotionPrediction:
    """HAR class label with softmax confidence."""

    motion_class: str
    confidence: float
    probabilities: dict[str, float]


class MotionHarModel:
    """Load WISDM-trained RF or fall back to threshold rules."""

    def __init__(self, model_path: Path | None = None) -> None:
        from biofizic.paths import models_dir

        path = model_path or (models_dir() / "motion_har_wisdm.joblib")
        self._available = path.exists()
        self._clf = None
        self._classes: list[str] = list(MOTION_CLASSES)
        if self._available:
            bundle = joblib.load(path)
            self._clf = bundle["model"]
            self._classes = list(bundle.get("classes", self._classes))
        self.calibrator = MotionCalibrator()

    @property
    def available(self) -> bool:
        return self._available and self._clf is not None

    def predict(self, feat: MotionFeatureVector) -> MotionPrediction:
        raw = feat.values
        if not self.available:
            return self._fallback(raw)

        x = self.calibrator.normalize(raw).reshape(1, -1)
        proba = self._clf.predict_proba(x)[0]
        idx = int(np.argmax(proba))
        label = self._classes[idx]
        self.calibrator.observe(raw, label)
        return MotionPrediction(
            motion_class=label,
            confidence=float(proba[idx]),
            probabilities={c: float(p) for c, p in zip(self._classes, proba)},
        )

    def _fallback(self, raw: np.ndarray) -> MotionPrediction:
        acc_p90 = raw[1] if len(raw) > 1 else raw[0]
        gyro_p90 = raw[4] if len(raw) > 4 else 0.0
        if acc_p90 >= 1.45 and gyro_p90 < 2.0:
            label = "WALK"
        elif acc_p90 >= 0.35 or gyro_p90 >= 0.5:
            label = "SCROLL"
        else:
            label = "STILL"
        self.calibrator.observe(raw, label)
        return MotionPrediction(
            motion_class=label,
            confidence=0.55,
            probabilities={label: 0.55},
        )
