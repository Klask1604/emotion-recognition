"""Personal valence baseline: the neutral point is the subject's MEASURED resting
median (not assumed 0), so the cross-device WESAD bias is absorbed and a shift
relative to the subject's own rest is what reads as positive/negative valence."""

from __future__ import annotations

from affectus.dsp.valence_baseline import ValenceBaselineStore, ValenceSmoother


def _calibrate(store, value, n=10):
    # Feed n spaced resting samples at a given valence_z (simulate a subject whose
    # resting valence_z sits at `value` due to cross-device bias).
    for i in range(n):
        store.observe_resting(value + (0.01 if i % 2 else -0.01), now=float(i * 5))


def test_locks_after_min_samples():
    s = ValenceBaselineStore()
    assert not s.is_ready
    _calibrate(s, -0.55)
    assert s.is_ready
    # neutral is the subject's measured median, ~-0.55, NOT 0.
    assert -0.6 < s.neutral_valence_z < -0.5


def test_biased_rest_maps_to_zero_personal():
    # A subject pinned at valence_z=-0.55 by domain shift should read ~0 personal
    # valence at rest (neutral), not "very negative".
    s = ValenceBaselineStore()
    _calibrate(s, -0.55)
    assert abs(s.valence_personal(-0.55)) < 0.5  # near personal neutral


def test_shift_toward_positive_reads_positive():
    # After calibrating at -0.55, a sample at -0.20 (less negative than the
    # subject's rest) is MORE positive than their neutral -> personal v > 0.
    s = ValenceBaselineStore()
    _calibrate(s, -0.55)
    assert s.valence_personal(-0.20) > 0.0


def test_shift_toward_negative_reads_negative():
    s = ValenceBaselineStore()
    _calibrate(s, -0.55)
    assert s.valence_personal(-0.90) < 0.0


def test_zero_before_lock():
    s = ValenceBaselineStore()
    s.observe_resting(-0.55, now=0.0)
    assert not s.is_ready
    assert s.valence_personal(-0.20) == 0.0  # no verdict until calibrated


def test_reset_clears():
    s = ValenceBaselineStore()
    _calibrate(s, -0.55)
    assert s.is_ready
    s.reset()
    assert not s.is_ready
    assert s.neutral_valence_z is None


def test_smoother_averages_over_window():
    sm = ValenceSmoother(window_s=30.0)
    # A spike at t=0 then steady ~0 should be damped, not reported at full height.
    assert sm.update(2.0, now=0.0) == 2.0          # first sample = itself
    sm.update(0.0, now=1.0)
    out = sm.update(0.0, now=2.0)
    assert -0.1 < out < 1.0                          # spike diluted by the zeros


def test_smoother_drops_samples_outside_window():
    sm = ValenceSmoother(window_s=30.0)
    sm.update(1.0, now=0.0)
    # 40 s later the old sample is outside the 30 s window -> mean is just the new.
    out = sm.update(-1.0, now=40.0)
    assert out == -1.0


def test_smoother_steady_state():
    sm = ValenceSmoother(window_s=30.0)
    for i in range(30):
        out = sm.update(0.5, now=float(i))
    assert abs(out - 0.5) < 1e-9                      # constant input -> constant mean


def test_deadband_stable_and_in_personal_z_units():
    # Dead-band is a fixed fraction of the (constant, by construction) resting-z
    # spread, so it is the SAME value run-to-run and before/after calibration —
    # no drift, unlike the old volatile in-memory smoother window.
    s = ValenceBaselineStore()
    db_before = s.deadband()
    _calibrate(s, -0.55)
    db_after = s.deadband()
    assert db_before == db_after                       # stable across lock
    assert 0.1 < db_after < 0.4                         # ~0.22, sane neutral half-width


def test_deadband_scales_with_reactivity():
    s = ValenceBaselineStore()
    base = s.deadband(reactivity_scale=1.0)
    assert s.deadband(reactivity_scale=0.6) < base      # low-responder -> narrower
    assert s.deadband(reactivity_scale=1.4) > base      # expressive -> wider
    assert s.deadband() == base                          # default signature unchanged


def test_reported_valence_persists(tmp_path):
    path = tmp_path / "valence_baseline.json"
    a = ValenceBaselineStore(path=path)
    a.reset_for_recalibration(reported_valence=0.6)
    b = ValenceBaselineStore(path=path)
    assert b.reported_baseline_valence == 0.6


def test_polarity_inverted_flips_sign():
    # Subject self-reports PLEASANT (+0.6) but the model's resting valence sits
    # strongly NEGATIVE -> polarity inverted -> personal valence sign flips.
    s = ValenceBaselineStore()
    s.reset_for_recalibration(reported_valence=0.6)
    _calibrate(s, -0.55)                                 # model resting median ~-0.55
    assert s._polarity_inverted
    # A raw value more positive than the median would normally read positive;
    # with inversion it reads negative (and vice versa).
    assert s.valence_personal(-0.20) < 0.0


def test_no_polarity_flip_when_consistent():
    s = ValenceBaselineStore()
    s.reset_for_recalibration(reported_valence=-0.6)     # reports unpleasant
    _calibrate(s, -0.55)                                 # model agrees (negative)
    assert not s._polarity_inverted
