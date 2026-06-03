"""
WESAD-trained valence fusion channel.

The first ML model to enter the production decision. Trained on WESAD wrist BVP
(stress vs amusement = the valence axis, ~85% subject-independent with the
unified PPG feature vector), saved as models/valence_wesad.joblib by
train/train_valence_wesad.py. Uses the SAME extractor as training
(legacy.valence_features.extract_valence_feature_vector) for train/serve parity.

Mirrors the temperature-channel pattern: a lazily-loaded state object plus an
evaluate function returning a FusionChannel (or None). Bounded by
VALENCE_WESAD_MAX_WEIGHT (default 0 = disabled) since it is a cross-device
domain-shift model — it must only nudge, never dominate. Requires the raw PPG
waveform (vascular/morphology features), so its capability set includes PPG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from affectus.config import VALENCE_WESAD_MAX_WEIGHT

log = logging.getLogger("valence_wesad_channel")


def default_model_path() -> Path:
    # this file: affectus/engine/channels/valence_wesad.py -> project root is 4 up
    return Path(__file__).resolve().parents[3] / "models" / "valence_wesad.joblib"


@dataclass
class ValenceWesadState:
    """Holds the loaded model bundle (or None if unavailable). Constructed once
    by the pipeline; absence of the model file degrades gracefully to a no-op."""

    model: object | None = None
    feature_names: list | None = None
    feature_mean: object | None = None
    feature_std: object | None = None
    _loaded: bool = False

    def ensure_loaded(self, path: Path | None = None) -> None:
        if self._loaded:
            return
        self._loaded = True
        p = path or default_model_path()
        if not p.exists():
            log.info("valence_wesad model not found at %s — channel disabled", p)
            return
        try:
            import joblib
            bundle = joblib.load(p)
            self.model = bundle["model"]
            self.feature_names = bundle["feature_names"]
            self.feature_mean = np.asarray(bundle["feature_mean"], dtype=float)
            self.feature_std = np.asarray(bundle["feature_std"], dtype=float)
        except Exception as exc:  # noqa: BLE001
            log.warning("valence_wesad model load failed: %s", exc)
            self.model = None


@dataclass(frozen=True)
class ValenceWesadResult:
    z: float            # >0 = positive valence, <0 = negative (centred at 0)
    confidence: float   # [0, 1]; probability margin |2p-1|
    p_positive: float


def evaluate_valence_wesad(
    state: ValenceWesadState, green: list[int], timestamps_ms: list[int]
) -> ValenceWesadResult | None:
    """Predict valence from one PPG window. Returns None when the model is absent
    or the features cannot be computed (so the channel is skipped, not zeroed)."""
    state.ensure_loaded()
    if state.model is None:
        return None
    from affectus.research.valence.features import extract_valence_feature_vector

    vec = extract_valence_feature_vector(green, timestamps_ms)
    if vec is None:
        return None
    x = (np.asarray(vec, dtype=float) - state.feature_mean) / state.feature_std
    try:
        proba = state.model.predict_proba(x.reshape(1, -1))[0]
    except Exception:  # noqa: BLE001
        return None
    p_pos = float(proba[1])               # class 1 = positive valence
    z = 2.0 * p_pos - 1.0                 # map [0,1] -> [-1, 1]
    confidence = abs(z)                   # probability margin
    return ValenceWesadResult(z=z, confidence=confidence, p_positive=p_pos)


def valence_weight(confidence: float) -> float:
    """Capped fusion weight: confidence · cap. Zero when the channel is disabled
    (VALENCE_WESAD_MAX_WEIGHT == 0), making the whole channel an exact no-op."""
    return confidence * VALENCE_WESAD_MAX_WEIGHT
