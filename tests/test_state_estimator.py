"""Kalman stress-state estimator: quality-weighted smoothing + hold on low quality.

The Kalman lives inside `biofizic.engine.decision` now (consolidated with the
CUSUM and the population/personal gate). These tests still cover the same
behaviour by driving the internal `_kalman_update` directly — they remain
unit tests of the smoother, not of the full pipeline.
"""

from __future__ import annotations

from biofizic.engine.decision import DecisionState, _kalman_update


def test_high_quality_epoch_moves_estimate_more_than_low_quality():
    good = DecisionState()
    bad = DecisionState()
    gx, ggain = _kalman_update(good, 2.0, quality=1.0)
    bx, bgain = _kalman_update(bad, 2.0, quality=0.0)
    assert ggain > bgain
    assert gx > bx  # the high-quality estimate moved further toward the measurement


def test_low_quality_epoch_barely_moves_estimate():
    est = DecisionState()
    _kalman_update(est, 1.0, quality=1.0)  # establish a value
    before = est.estimator_x
    _kalman_update(est, 5.0, quality=0.0)  # garbage epoch
    assert abs(est.estimator_x - before) < 0.2  # held, not whipped to 5


def test_sustained_signal_converges_toward_measurement():
    est = DecisionState()
    for _ in range(30):
        _kalman_update(est, 2.0, quality=1.0)
    assert est.estimator_x > 1.5  # converges toward the persistent measurement


def test_single_spike_is_attenuated():
    est = DecisionState()
    for _ in range(20):
        _kalman_update(est, 0.0, quality=1.0)
    _kalman_update(est, 4.0, quality=1.0)  # one-epoch spike
    assert est.estimator_x < 2.0  # not adopted wholesale in a single step
