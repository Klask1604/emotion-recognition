"""Tests for the respiration comparator legacy engine (B2c).

The comparator runs both estimators on one batch and reports both, plus an
agreement metric when both are confident. It must also degrade gracefully when
only one source is available (e.g. no raw PPG).
"""

from __future__ import annotations

import math

import numpy as np

from biofizic.ingestion.messages import AcquisitionBatchMessage
from biofizic.legacy.respiration_compare import RespirationCompareEngine


def _ibi_modulated(breathing_hz: float, n: int = 80):
    rng = np.random.default_rng(0)
    ms, ts = [], []
    t = 0.0
    for _ in range(n):
        rr = 850 + 40 * math.sin(2 * math.pi * breathing_hz * t) + rng.normal(0, 2)
        rr = max(300.0, rr)
        t += rr / 1000.0
        ms.append(int(round(rr)))
        ts.append(int(round(t * 1000)))
    return ms, ts


def _batch(ibi_ms, ibi_ts, ppg_green=None, ppg_ts=None) -> AcquisitionBatchMessage:
    return AcquisitionBatchMessage(
        timestamp_publish_ms=ibi_ts[-1] if ibi_ts else 0,
        timestamp_anchor_ms=ibi_ts[-1] if ibi_ts else 0,
        sequence=1,
        ibi_intervals_ms=ibi_ms,
        ibi_timestamps_ms=ibi_ts,
        ppg_green=ppg_green or [],
        ppg_timestamps_ms=ppg_ts or [],
    )


def test_rsa_arm_runs_from_ibi_only():
    ms, ts = _ibi_modulated(0.25)
    out = RespirationCompareEngine().compute(_batch(ms, ts))
    assert out is not None
    assert "rsa_bpm" in out and "rsa_conf" in out
    # No PPG provided -> no PPG arm, no agreement metric.
    assert "ppg_bpm" not in out
    assert "agree_bpm_diff" not in out


def test_returns_none_when_no_data():
    out = RespirationCompareEngine().compute(_batch([], []))
    assert out is None


def test_both_arms_and_agreement_when_both_present():
    ms, ts = _ibi_modulated(0.25)
    # Synthetic PPG amplitude-modulated at the same 0.25 Hz.
    fs, seconds = 25.0, 40.0
    n = int(fs * seconds)
    green, pts = [], []
    rng = np.random.default_rng(2)
    for i in range(n):
        t = i / fs
        amp = 1.0 + 0.5 * math.sin(2 * math.pi * 0.25 * t)
        pulse = amp * math.sin(2 * math.pi * 1.2 * t)
        green.append(int(round(2000 + 500 * (pulse + rng.normal(0, 0.02)))))
        pts.append(int(round(t * 1000)))
    out = RespirationCompareEngine().compute(_batch(ms, ts, green, pts))
    assert out is not None
    assert "rsa_bpm" in out and "ppg_bpm" in out
    # Both confident on a clean 15 br/min signal -> agreement reported.
    assert "agree_bpm_diff" in out


def test_comparator_never_feeds_production_decision():
    """The comparator is a legacy/research engine: its output is a plain dict on
    its own topic and must never appear in the production PhysiologyDecision.
    (The toggle itself may be flipped on for a research session; what matters is
    that it stays on the legacy path.)"""
    from biofizic.legacy import LegacyEngines
    # Whatever the toggle, the comparator output is surfaced only via
    # LegacyOutputs.respiration, never via the decision path.
    assert hasattr(LegacyEngines, "run")
    from dataclasses import fields
    from biofizic.compute_features.results import PhysiologyDecision
    decision_fields = {f.name for f in fields(PhysiologyDecision)}
    assert "respiration" not in decision_fields
    assert "rsa_bpm" not in decision_fields
