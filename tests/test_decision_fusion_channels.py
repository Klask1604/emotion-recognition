"""Unit tests for the multi-channel fusion primitive (engine/decision._fuse).

B0 generalised the hard-wired two-term blend
    z_fused = Q·z_hrv + (1-Q)·z_hr
into a weighted mean over a list of channels, so temp/resp channels can be
appended later. These tests pin two contracts:

  1. With only HRV (weight Q) and HR (weight 1-Q) the weighted mean is exactly
     the old two-term blend — the denominator is 1, so nothing changes.
  2. A third channel with weight 0 is an exact no-op (this is how temp/resp
     enter before their quality gate is wired in).
  3. A third channel with non-zero weight re-normalises correctly.
"""

from __future__ import annotations

import pytest

from biofizic.engine.decision import FusionChannel, _fuse


def _old_blend(q: float, z_hrv: float, z_hr: float) -> float:
    """The pre-B0 formula, reproduced here as the golden reference."""
    return q * z_hrv + (1.0 - q) * z_hr


@pytest.mark.parametrize(
    "q, z_hrv, z_hr",
    [
        (0.9, 1.2, -0.3),
        (0.03, 0.0, 2.5),   # motion: HRV collapsed, HR carries it
        (0.5, 1.0, 1.0),
        (1.0, 2.0, 0.0),    # perfect HRV: HR contributes nothing
        (0.0, 5.0, 0.4),    # no HRV weight: pure HR
    ],
)
def test_two_channel_matches_old_blend(q: float, z_hrv: float, z_hr: float):
    channels = [
        FusionChannel("hrv", z=z_hrv, weight=q, confidence=q),
        FusionChannel("hr", z=z_hr, weight=1.0 - q, confidence=0.7),
    ]
    z, _ = _fuse(channels)
    assert z == pytest.approx(_old_blend(q, z_hrv, z_hr), abs=1e-12)


def test_zero_weight_channel_is_a_noop():
    """A temp/resp channel entering with weight 0 must not change z_fused —
    this is the safety guarantee for adding channels before gating them."""
    base = [
        FusionChannel("hrv", z=1.0, weight=0.8, confidence=0.8),
        FusionChannel("hr", z=-0.5, weight=0.2, confidence=0.7),
    ]
    z_base, conf_base = _fuse(base)
    with_dead_channel = base + [
        FusionChannel("temp", z=3.0, weight=0.0, confidence=0.0),
    ]
    z_new, conf_new = _fuse(with_dead_channel)
    assert z_new == pytest.approx(z_base, abs=1e-12)
    assert conf_new == pytest.approx(conf_base, abs=1e-12)


def test_third_channel_renormalises():
    """With three weighted channels the denominator is the weight sum, so the
    result is the correctly re-normalised weighted mean."""
    channels = [
        FusionChannel("hrv", z=1.0, weight=0.5, confidence=0.5),
        FusionChannel("hr", z=0.0, weight=0.3, confidence=0.7),
        FusionChannel("temp", z=2.0, weight=0.2, confidence=0.4),
    ]
    z, conf = _fuse(channels)
    expected_z = (0.5 * 1.0 + 0.3 * 0.0 + 0.2 * 2.0) / (0.5 + 0.3 + 0.2)
    expected_conf = (0.5 * 0.5 + 0.3 * 0.7 + 0.2 * 0.4) / 1.0
    assert z == pytest.approx(expected_z, abs=1e-12)
    assert conf == pytest.approx(expected_conf, abs=1e-12)


def test_no_weight_returns_zero():
    """All channels dead → (0, 0), not a division by zero."""
    channels = [
        FusionChannel("hrv", z=1.0, weight=0.0, confidence=0.0),
        FusionChannel("hr", z=2.0, weight=0.0, confidence=0.0),
    ]
    assert _fuse(channels) == (0.0, 0.0)
