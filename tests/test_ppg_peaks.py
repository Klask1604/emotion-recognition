"""PPG peak detection (research): finds pulse peaks on a synthetic wave."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("scipy", reason="scipy is a research-only dependency")

from biofizic.dsp.ppg_peaks import detect_ppg_peaks


def test_detects_pulse_peaks_on_synthetic_wave():
    fs, dur, f = 25.0, 8.0, 1.2  # 1.2 Hz pulse == 72 bpm
    n = int(fs * dur)
    ts = [int(i * 1000 / fs) for i in range(n)]
    green = [int(1000 + 200 * np.sin(2 * np.pi * f * i / fs)) for i in range(n)]

    res = detect_ppg_peaks(green, ts)

    assert res.n_peaks >= int(f * dur) - 2  # ~9-10 peaks expected
    assert res.ppa > 0
    assert abs(res.sample_rate_hz - fs) < 1.0
    if res.reconstructed_ibi_ms:
        assert abs(float(np.mean(res.reconstructed_ibi_ms)) - 1000.0 / f) < 120


def test_too_few_samples_returns_empty():
    res = detect_ppg_peaks([1000, 1010, 990], [0, 40, 80])
    assert res.n_peaks == 0
    assert res.ppa == 0.0
