"""Isolated unit tests for the RSA-from-IBI respiration estimator (B2a).

Synthetic IBI series are built by modulating a baseline RR interval at a known
breathing frequency, so we can check the estimator recovers it AND that the
confidence gate behaves honestly at the slow / ambiguous end (the regime the
target user actually breathes in).
"""

from __future__ import annotations

import math

import numpy as np

from biofizic.config import RSA_MIN_BEATS
from biofizic.engine.channels.respiration_rsa import estimate_respiration_rsa
from biofizic.ingestion.messages import InterbeatIntervalEntry


def _synth_ibi(
    breathing_hz: float,
    *,
    mean_rr_ms: float = 850.0,
    rsa_amplitude_ms: float = 40.0,
    n_beats: int = 80,
    noise_ms: float = 2.0,
    seed: int = 0,
) -> list[InterbeatIntervalEntry]:
    """Build an IBI series whose RR oscillates at breathing_hz (RSA)."""
    rng = np.random.default_rng(seed)
    entries: list[InterbeatIntervalEntry] = []
    t_ms = 0.0
    for _ in range(n_beats):
        phase = 2.0 * math.pi * breathing_hz * (t_ms / 1000.0)
        rr = mean_rr_ms + rsa_amplitude_ms * math.sin(phase) + rng.normal(0, noise_ms)
        rr = max(300.0, rr)
        t_ms += rr
        entries.append(InterbeatIntervalEntry(interval_ms=int(round(rr)), timestamp_ms=int(t_ms)))
    return entries


def test_recovers_normal_breathing_rate():
    # 0.25 Hz = 15 breaths/min — squarely in the RSA band.
    entries = _synth_ibi(breathing_hz=0.25)
    est = estimate_respiration_rsa(entries)
    assert est.confidence > 0.0
    assert est.breaths_per_min == _approx_bpm(15.0, tol=3.0), est.breaths_per_min


def test_recovers_faster_breathing_rate():
    # 0.33 Hz ≈ 20 breaths/min.
    entries = _synth_ibi(breathing_hz=0.33)
    est = estimate_respiration_rsa(entries)
    assert est.confidence > 0.0
    assert est.breaths_per_min == _approx_bpm(20.0, tol=3.0), est.breaths_per_min


def test_slow_breathing_confidence_collapses():
    # 0.12 Hz ≈ 7 breaths/min — below the slow edge, ambiguous with Mayer waves.
    # The estimator may still find a peak, but confidence must be ~0 (honest).
    entries = _synth_ibi(breathing_hz=0.12)
    est = estimate_respiration_rsa(entries)
    assert est.confidence == 0.0


def test_too_few_beats_returns_zero_confidence():
    entries = _synth_ibi(breathing_hz=0.25, n_beats=RSA_MIN_BEATS - 1)
    est = estimate_respiration_rsa(entries)
    assert est.confidence == 0.0
    assert est.breaths_per_min == 0.0


def test_flat_series_has_no_peak():
    # No RSA at all (constant RR) -> no dominant peak -> confidence 0.
    entries = [
        InterbeatIntervalEntry(interval_ms=850, timestamp_ms=850 * i)
        for i in range(1, 81)
    ]
    est = estimate_respiration_rsa(entries)
    assert est.confidence == 0.0


def test_empty_input_is_safe():
    est = estimate_respiration_rsa([])
    assert est.confidence == 0.0
    assert est.breaths_per_min == 0.0


class _ApproxBpm:
    def __init__(self, target: float, tol: float):
        self.target = target
        self.tol = tol

    def __eq__(self, other: object) -> bool:
        return isinstance(other, (int, float)) and abs(other - self.target) <= self.tol

    def __repr__(self) -> str:
        return f"~{self.target}±{self.tol} bpm"


def _approx_bpm(target: float, tol: float) -> _ApproxBpm:
    return _ApproxBpm(target, tol)
