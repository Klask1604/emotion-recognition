"""ECG R-peak detection: clean QRS -> correct IBI; a lead-off gap fabricates no
giant IBI; too-short / all-lead-off windows return empty."""

from __future__ import annotations

import math

import pytest

pytest.importorskip("scipy")

from affectus.shared.dsp.ecg_peaks import detect_ecg_rpeaks


def _synth_ecg(seconds=20.0, fs=500, bpm=75, lead_off_window=None):
    """Synthetic ECG: a sharp Gaussian QRS bump every RR interval. Optionally
    force lead_off=1 over a (start_s, end_s) window."""
    rr_s = 60.0 / bpm
    n = int(seconds * fs)
    ts = [int(i * 1000 / fs) for i in range(n)]
    ecg = [0.0] * n
    # place QRS bumps
    t = rr_s
    while t < seconds:
        center = int(t * fs)
        for k in range(-6, 7):
            idx = center + k
            if 0 <= idx < n:
                ecg[idx] += 1000.0 * math.exp(-(k * k) / 4.0)  # ~12 ms QRS
        t += rr_s
    lead = [0] * n
    if lead_off_window is not None:
        a, b = lead_off_window
        for i in range(int(a * fs), int(b * fs)):
            if 0 <= i < n:
                lead[i] = 1
    return ecg, ts, lead


def test_clean_ecg_recovers_correct_ibi():
    ecg, ts, lead = _synth_ecg(seconds=20, bpm=75)
    res = detect_ecg_rpeaks(ecg, ts, lead)
    assert res.n_peaks > 15
    mean_ibi = sum(res.reconstructed_ibi_ms) / len(res.reconstructed_ibi_ms)
    assert 760 <= mean_ibi <= 840          # 75 bpm == 800 ms, allow tolerance
    assert res.lead_on_fraction == 1.0


def test_lead_off_gap_makes_no_giant_ibi():
    # Lead off for the middle 4 s. No reconstructed IBI should span that gap.
    ecg, ts, lead = _synth_ecg(seconds=20, bpm=75, lead_off_window=(8.0, 12.0))
    res = detect_ecg_rpeaks(ecg, ts, lead)
    assert res.reconstructed_ibi_ms                      # still got beats either side
    assert max(res.reconstructed_ibi_ms) <= 2000         # none spans the 4 s gap
    assert res.lead_on_fraction < 1.0


def test_all_lead_off_returns_empty():
    ecg, ts, _ = _synth_ecg(seconds=20, bpm=75)
    lead = [1] * len(ecg)
    res = detect_ecg_rpeaks(ecg, ts, lead)
    assert res.reconstructed_ibi_ms == []


def test_too_few_samples_returns_empty():
    res = detect_ecg_rpeaks([0, 1, 2], [0, 2, 4], None)
    assert res.n_peaks == 0
    assert res.reconstructed_ibi_ms == []
