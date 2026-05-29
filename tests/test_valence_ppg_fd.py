"""Isolated tests for the PPG frequency-domain valence features.

Synthetic PPG with KNOWN harmonic content (controlled amplitudes at f0, 2f0,
3f0) lets us check the extractor recovers the right band powers and ratios, at
both 25 Hz (continuous) and 100 Hz (on-demand) — the method must be sample-rate
agnostic since it derives fs from timestamps.
"""

from __future__ import annotations

import math

import numpy as np

from biofizic.config import VALENCE_FD_MIN_SAMPLES
from biofizic.legacy.valence_ppg_fd import extract_valence_fd_features


def _synth_ppg(hr_bpm: float, fs: float, seconds: float,
               a1: float, a2: float, a3: float, noise: float = 0.01, seed: int = 0):
    """PPG-like signal = a1*sin(2pi f0 t) + a2*sin(2pi 2f0 t) + a3*sin(2pi 3f0 t)."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    f0 = hr_bpm / 60.0
    green, ts = [], []
    for i in range(n):
        t = i / fs
        v = (a1 * math.sin(2 * math.pi * f0 * t)
             + a2 * math.sin(2 * math.pi * 2 * f0 * t)
             + a3 * math.sin(2 * math.pi * 3 * f0 * t)
             + rng.normal(0, noise))
        green.append(int(round(2000 + 500 * v)))
        ts.append(int(round(t * 1000)))
    return green, ts


def test_recovers_dominant_fundamental_at_25hz():
    # Strong fundamental, weak harmonics -> BFn should dominate.
    green, ts = _synth_ppg(hr_bpm=84, fs=25.0, seconds=20, a1=1.0, a2=0.3, a3=0.1)
    f = extract_valence_fd_features(green, ts, hr_bpm=84)
    assert f.valid
    assert f.bf_n > f.fhf_n > f.shf_n, (f.bf_n, f.fhf_n, f.shf_n)
    assert f.bf_n > 0.5


def test_recovers_dominant_fundamental_at_100hz():
    # Same content at 100 Hz must give the same qualitative ordering.
    green, ts = _synth_ppg(hr_bpm=84, fs=100.0, seconds=20, a1=1.0, a2=0.3, a3=0.1)
    f = extract_valence_fd_features(green, ts, hr_bpm=84)
    assert f.valid
    assert f.bf_n > f.fhf_n > f.shf_n
    assert f.bf_n > 0.5


def test_strong_first_harmonic_raises_fhf_ratio():
    # Boost the first harmonic; FHF/BF should rise vs a fundamental-only signal.
    weak, ts1 = _synth_ppg(hr_bpm=72, fs=100.0, seconds=20, a1=1.0, a2=0.1, a3=0.05)
    strong, ts2 = _synth_ppg(hr_bpm=72, fs=100.0, seconds=20, a1=1.0, a2=0.9, a3=0.05)
    fw = extract_valence_fd_features(weak, ts1, hr_bpm=72)
    fs = extract_valence_fd_features(strong, ts2, hr_bpm=72)
    assert fs.fhf_bf > fw.fhf_bf


def test_normalised_powers_sum_to_one():
    green, ts = _synth_ppg(hr_bpm=80, fs=100.0, seconds=20, a1=1.0, a2=0.5, a3=0.3)
    f = extract_valence_fd_features(green, ts, hr_bpm=80)
    assert f.valid
    assert abs((f.bf_n + f.fhf_n + f.shf_n) - 1.0) < 1e-6


def test_invalid_without_hr():
    green, ts = _synth_ppg(hr_bpm=80, fs=100.0, seconds=20, a1=1.0, a2=0.5, a3=0.3)
    f = extract_valence_fd_features(green, ts, hr_bpm=0)
    assert not f.valid


def test_invalid_too_few_samples():
    green, ts = _synth_ppg(hr_bpm=80, fs=25.0, seconds=2, a1=1.0, a2=0.5, a3=0.3)
    assert len(green) < VALENCE_FD_MIN_SAMPLES
    f = extract_valence_fd_features(green, ts, hr_bpm=80)
    assert not f.valid


def test_empty_safe():
    f = extract_valence_fd_features([], [], hr_bpm=80)
    assert not f.valid
    assert f.as_dict()["bf"] == 0.0
