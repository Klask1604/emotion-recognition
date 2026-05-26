"""
WESAD RandomForest stress probability — parallel research engine.

DOMAIN SHIFT (the whole point): WESAD was recorded with a RespiBAN chest ECG
(700 Hz) and an Empatica E4 wrist (64 Hz BVP), NOT a Galaxy Watch 7. A model
trained on it is expected to transfer poorly to GW7 wrist PPG; the comparison
"deterministic personal-baseline vs ML-trained-on-a-foreign-dataset" is exactly
the negative result we want to show with data. Published on
`biofizic/legacy/wesad`, never on `biofizic/state`.

Train/serve feature parity: both training (train/train_wesad.py) and this engine
build the feature vector from our own HrvMetrics via wesad_feature_vector(), so
there is no FFT-vs-linear style skew (the lesson from the retired WISDM HAR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from biofizic.compute_features.results import HrvMetrics

WESAD_FEATURE_NAMES = [
    "rmssd_ms",
    "sdnn_ms",
    "pnn50_percent",
    "kubios_stress_index",
    "mean_heart_rate_bpm",
]


def wesad_feature_vector(m: HrvMetrics) -> list[float]:
    return [
        m.rmssd_ms,
        m.sdnn_ms,
        m.pnn50_percent,
        m.kubios_stress_index,
        m.mean_heart_rate_bpm,
    ]


def default_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "wesad_rf.joblib"


class WesadEngine:
    def __init__(self, path: Path | None = None) -> None:
        import joblib  # lazy, research-only

        p = path or default_model_path()
        if not p.exists():
            raise FileNotFoundError(
                f"WESAD model not found at {p}. Train it with train/train_wesad.py "
                f"or turn ENABLE_WESAD off."
            )
        bundle = joblib.load(p)
        self._clf = bundle["model"]
        self._mean = np.asarray(bundle["feature_mean"], dtype=float)
        self._std = np.asarray(bundle["feature_std"], dtype=float)
        classes = list(bundle["classes"])
        self._stress_idx = classes.index("stress") if "stress" in classes else int(np.argmax(classes))

    def predict(self, metrics: HrvMetrics) -> dict:
        x = (np.asarray(wesad_feature_vector(metrics), dtype=float) - self._mean) / self._std
        proba = self._clf.predict_proba(x.reshape(1, -1))[0]
        return {"p_stress": round(float(proba[self._stress_idx]), 3)}
