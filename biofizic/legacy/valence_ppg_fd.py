"""
PPG frequency-domain features for valence — replication of the method in
Frontiers Physiol. 2025 ("An emotion recognition method based on frequency-
domain features of PPG", PMC11893849), adapted to a Galaxy Watch consumer PPG
stream (~25 Hz vs the paper's 125 Hz).

The paper extracts nine features from the FFT of the band-passed PPG waveform,
built from the power in ±0.2 Hz bands centred on the heart-rate fundamental and
its first two harmonics:

    BF   = power in [f0-0.2, f0+0.2]            (fundamental, f0 = HR/60)
    FHF  = power in [2*f0-0.2, 2*f0+0.2]        (first harmonic)
    SHF  = power in [3*f0-0.2, 3*f0+0.2]        (second harmonic)
    BFn  = BF/(BF+FHF+SHF), FHFn, SHFn          (normalised powers)
    FHF/BF, SHF/BF, SHF/FHF                     (ratios)

This module ONLY extracts the features (pure function + tests). It does NOT
classify: a validated valence verdict needs a trained SVM on labelled data,
which we do not have for the watch. The features are published on
biofizic/legacy/valence_fd so the thesis can (a) show they are computable on
consumer PPG and (b) later train/評価 a classifier against ground-truth. This
is the honest replacement for the ad-hoc RMSSD/PPA valence heuristic.

NOTE ON HARDWARE: at ~25 Hz the Nyquist is 12.5 Hz; the third harmonic of a
typical HR (3*1.45 = 4.4 Hz) is well within range, so all three bands are
resolvable — but the spectral resolution is coarser than the paper's 125 Hz.
This is a documented consumer-hardware adaptation, not the original method.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biofizic.config import (
    VALENCE_FD_BAND_HALFWIDTH_HZ,
    VALENCE_FD_MIN_SAMPLES,
    VALENCE_FD_PPG_BAND_HI_HZ,
    VALENCE_FD_PPG_BAND_LO_HZ,
)


@dataclass(frozen=True)
class ValenceFdFeatures:
    """The nine frequency-domain features, plus diagnostics. All powers are in
    arbitrary (normalised-FFT) units; the ratios/normalised powers are the
    scale-free, comparable quantities."""

    bf: float
    fhf: float
    shf: float
    bf_n: float
    fhf_n: float
    shf_n: float
    fhf_bf: float
    shf_bf: float
    shf_fhf: float
    f0_hz: float
    valid: bool

    def as_dict(self) -> dict:
        return {
            "bf": round(self.bf, 4),
            "fhf": round(self.fhf, 4),
            "shf": round(self.shf, 4),
            "bf_n": round(self.bf_n, 4),
            "fhf_n": round(self.fhf_n, 4),
            "shf_n": round(self.shf_n, 4),
            "fhf_bf": round(self.fhf_bf, 4),
            "shf_bf": round(self.shf_bf, 4),
            "shf_fhf": round(self.shf_fhf, 4),
            "f0_hz": round(self.f0_hz, 3),
        }


def _bandpass(sig: np.ndarray, fs: float) -> np.ndarray:
    """1–N Hz band-pass (N clipped below Nyquist). Returns zero-mean filtered
    signal, or the detrended raw signal if the filter cannot be built."""
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    hi = min(VALENCE_FD_PPG_BAND_HI_HZ, nyq * 0.95)
    lo = VALENCE_FD_PPG_BAND_LO_HZ
    centered = sig - float(np.mean(sig))
    if hi <= lo or fs <= 0:
        return centered
    b, a = butter(2, [lo / nyq, hi / nyq], btype="band")
    if len(centered) <= 3 * max(len(a), len(b)):
        return centered
    return filtfilt(b, a, centered)


def _band_power(freqs: np.ndarray, psd: np.ndarray, centre: float, half: float) -> float:
    """Sum of PSD in [centre-half, centre+half]."""
    if centre <= 0:
        return 0.0
    mask = (freqs >= centre - half) & (freqs <= centre + half)
    if not np.any(mask):
        return 0.0
    return float(np.sum(psd[mask]))


def extract_valence_fd_features(
    green: list[int], timestamps_ms: list[int], hr_bpm: float
) -> ValenceFdFeatures:
    """Extract the nine PPG frequency-domain valence features.

    `hr_bpm` locates the fundamental f0 = HR/60; if it is unavailable (<=0) the
    features are invalid (we do not guess the fundamental from the spectrum here,
    to stay faithful to the paper which anchors the bands on the known HR)."""
    n = len(green)
    invalid = ValenceFdFeatures(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0, False)
    if n < VALENCE_FD_MIN_SAMPLES or len(timestamps_ms) != n or hr_bpm <= 0:
        return invalid

    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return invalid
    fs = (n - 1) / span_s
    if fs <= 0:
        return invalid

    sig = _bandpass(np.asarray(green, dtype=float), fs)

    # One-sided power spectrum.
    fft = np.fft.rfft(sig)
    psd = (np.abs(fft) ** 2) / n
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    f0 = hr_bpm / 60.0
    half = VALENCE_FD_BAND_HALFWIDTH_HZ
    bf = _band_power(freqs, psd, f0, half)
    fhf = _band_power(freqs, psd, 2.0 * f0, half)
    shf = _band_power(freqs, psd, 3.0 * f0, half)

    total = bf + fhf + shf
    if total <= 0:
        return invalid

    return ValenceFdFeatures(
        bf=bf,
        fhf=fhf,
        shf=shf,
        bf_n=bf / total,
        fhf_n=fhf / total,
        shf_n=shf / total,
        fhf_bf=fhf / bf if bf > 0 else 0.0,
        shf_bf=shf / bf if bf > 0 else 0.0,
        shf_fhf=shf / fhf if fhf > 0 else 0.0,
        f0_hz=f0,
        valid=True,
    )
