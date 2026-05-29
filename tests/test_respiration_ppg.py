"""Isolated unit tests for the PPG amplitude-modulation respiration estimator (B2b).

Synthetic PPG: a ~1.2 Hz pulse wave whose amplitude is modulated at a known
breathing frequency. The estimator should recover the breathing rate from the
amplitude envelope and gate confidence honestly at the slow edge.
"""

from __future__ import annotations

import math

import numpy as np

from biofizic.engine.channels.respiration_ppg import estimate_respiration_ppg


def _synth_ppg(
    breathing_hz: float,
    *,
    pulse_hz: float = 1.2,
    fs: float = 25.0,
    seconds: float = 40.0,
    mod_depth: float = 0.5,
    noise: float = 0.02,
    seed: int = 1,
):
    """A pulse wave at pulse_hz whose amplitude is modulated at breathing_hz."""
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    green = []
    ts = []
    for i in range(n):
        t = i / fs
        amplitude = 1.0 + mod_depth * math.sin(2.0 * math.pi * breathing_hz * t)
        pulse = amplitude * math.sin(2.0 * math.pi * pulse_hz * t)
        sample = pulse + rng.normal(0, noise)
        # PPG sensors report integers; scale up.
        green.append(int(round(2000 + 500 * sample)))
        ts.append(int(round(t * 1000)))
    return green, ts


def test_recovers_normal_breathing_rate_from_amplitude():
    # 0.25 Hz = 15 breaths/min modulation on the pulse amplitude.
    green, ts = _synth_ppg(breathing_hz=0.25)
    est = estimate_respiration_ppg(green, ts)
    assert est.confidence > 0.0
    assert abs(est.breaths_per_min - 15.0) <= 3.5, est.breaths_per_min


def test_recovers_faster_breathing_rate_from_amplitude():
    # 0.33 Hz ≈ 20 breaths/min.
    green, ts = _synth_ppg(breathing_hz=0.33)
    est = estimate_respiration_ppg(green, ts)
    assert est.confidence > 0.0
    assert abs(est.breaths_per_min - 20.0) <= 3.5, est.breaths_per_min


def test_slow_breathing_confidence_collapses():
    # 0.12 Hz ≈ 7 breaths/min — below the slow edge, confidence must be 0.
    green, ts = _synth_ppg(breathing_hz=0.12)
    est = estimate_respiration_ppg(green, ts)
    assert est.confidence == 0.0


def test_too_short_returns_zero():
    green, ts = _synth_ppg(breathing_hz=0.25, seconds=3.0)
    est = estimate_respiration_ppg(green, ts)
    assert est.confidence == 0.0
    assert est.breaths_per_min == 0.0


def test_empty_input_is_safe():
    est = estimate_respiration_ppg([], [])
    assert est.confidence == 0.0
    assert est.breaths_per_min == 0.0


def test_mismatched_lengths_safe():
    est = estimate_respiration_ppg([1, 2, 3], [10, 20])
    assert est.confidence == 0.0
