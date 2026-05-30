"""
Unified valence feature vector — the single extractor used by BOTH the WESAD
training script and the live valence fusion channel, guaranteeing train/serve
parity (the same feature order and definitions on both sides).

Combines the four PPG feature families that, on WESAD (stress vs amusement),
reach ~85% subject-independent valence accuracy:
  vascular  — perfusion index / tone (the strongest single family)
  freq      — HR-harmonic FFT features
  morph     — pulse-shape morphology
  hrv       — RMSSD / pNN50 / LF-HF (vagal tone)

extract_valence_feature_vector(green, ts) -> list[float] | None
Returns None when any family is invalid (too few beats / bad window), so callers
can skip the window. HR is estimated internally for the harmonic anchoring.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

from affectus.legacy.ppg_hrv_features import HRV_FEATURE_NAMES, extract_hrv_features
from affectus.legacy.ppg_vascular_features import (
    VASC_FEATURE_NAMES,
    extract_vascular_features,
)
from affectus.legacy.valence_ppg_fd import extract_valence_fd_features
from affectus.legacy.valence_ppg_morph import (
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)

_FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]

# Canonical feature order — identical in training and serving.
VALENCE_FEATURE_NAMES = (
    list(VASC_FEATURE_NAMES)
    + list(_FD_KEYS)
    + list(MORPH_FEATURE_NAMES)
    + list(HRV_FEATURE_NAMES)
)


def _estimate_hr(green: list[int], fs: float) -> float:
    sig = np.asarray(green, dtype=float)
    sig = sig - sig.mean()
    nyq = fs / 2.0
    if nyq <= 4.0:
        return 0.0
    b, a = butter(2, [0.5 / nyq, 4.0 / nyq], btype="band")
    if len(sig) <= 3 * max(len(a), len(b)):
        return 0.0
    filt = filtfilt(b, a, sig)
    pk, _ = find_peaks(filt, distance=int(0.4 * fs))
    dur = len(sig) / fs
    return len(pk) / dur * 60.0 if (dur > 0 and len(pk) > 1) else 0.0


def extract_valence_feature_vector(
    green: list[int], timestamps_ms: list[int]
) -> list[float] | None:
    """Return the unified valence feature vector, or None if any family fails."""
    n = len(green)
    if n < 64 or len(timestamps_ms) != n:
        return None
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return None
    fs = (n - 1) / span_s
    hr = _estimate_hr(green, fs)
    if hr <= 0:
        return None

    vasc = extract_vascular_features(green, timestamps_ms)
    fd = extract_valence_fd_features(green, timestamps_ms, hr_bpm=hr)
    morph = extract_morph_features(green, timestamps_ms)
    hrv = extract_hrv_features(green, timestamps_ms)
    if not (vasc.valid and fd.valid and morph.valid and hrv.valid):
        return None

    return (
        [vasc.as_dict()[k] for k in VASC_FEATURE_NAMES]
        + [fd.as_dict()[k] for k in _FD_KEYS]
        + [morph.as_dict()[k] for k in MORPH_FEATURE_NAMES]
        + [hrv.as_dict()[k] for k in HRV_FEATURE_NAMES]
    )
