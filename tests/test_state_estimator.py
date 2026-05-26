"""Kalman stress-state estimator: quality-weighted smoothing + hold on low quality."""

from __future__ import annotations

from biofizic.engine.state_estimator import StressStateEstimator


def test_high_quality_epoch_moves_estimate_more_than_low_quality():
    good = StressStateEstimator()
    bad = StressStateEstimator()
    gx, ggain = good.update(2.0, quality=1.0)
    bx, bgain = bad.update(2.0, quality=0.0)
    assert ggain > bgain
    assert gx > bx  # the high-quality estimate moved further toward the measurement


def test_low_quality_epoch_barely_moves_estimate():
    est = StressStateEstimator()
    est.update(1.0, quality=1.0)  # establish a value
    before = est.value()
    est.update(5.0, quality=0.0)  # garbage epoch
    assert abs(est.value() - before) < 0.2  # held, not whipped to 5


def test_sustained_signal_converges_toward_measurement():
    est = StressStateEstimator()
    for _ in range(30):
        est.update(2.0, quality=1.0)
    assert est.value() > 1.5  # converges toward the persistent measurement


def test_single_spike_is_attenuated():
    est = StressStateEstimator()
    for _ in range(20):
        est.update(0.0, quality=1.0)
    est.update(4.0, quality=1.0)  # one-epoch spike
    assert est.value() < 2.0  # not adopted wholesale in a single step
