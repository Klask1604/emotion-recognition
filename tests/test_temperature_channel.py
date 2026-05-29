"""Isolated unit tests for the skin-temperature arousal channel.

The channel is NOT yet wired into the pipeline; these tests pin its behaviour
in isolation (the B1a step) before integration:

  - baseline locks after the required resting epochs,
  - z is sign-inverted (colder skin => positive arousal z),
  - confidence is 0 before lock / on a missing sample,
  - ambient thermal drift erodes confidence (thermoregulation, not arousal),
  - a cold-start / drifting epoch contributes weight 0 to the fusion (no-op).
"""

from __future__ import annotations

import pytest

from biofizic.config import (
    TEMP_AMBIENT_DRIFT_C_FULL_PENALTY,
    TEMP_BASELINE_MIN_REST_EPOCHS,
)
from biofizic.engine.channels.temperature import (
    SkinTemperatureChannelState,
    evaluate_skin_temperature,
    skin_temperature_z,
)


def _locked_state(skin_c: float = 33.0, ambient_c: float = 24.0) -> SkinTemperatureChannelState:
    s = SkinTemperatureChannelState()
    # Slight natural jitter so MAD is non-zero and realistic.
    for i in range(TEMP_BASELINE_MIN_REST_EPOCHS):
        jitter = 0.1 if i % 2 else -0.1
        s.observe_resting(skin_temp_c=skin_c + jitter, ambient_temp_c=ambient_c)
    assert s.is_ready
    return s


def test_baseline_locks_after_min_epochs():
    s = SkinTemperatureChannelState()
    for i in range(TEMP_BASELINE_MIN_REST_EPOCHS - 1):
        s.observe_resting(skin_temp_c=33.0)
        assert not s.is_ready
    s.observe_resting(skin_temp_c=33.0)
    assert s.is_ready
    assert s.baseline_skin_c == pytest.approx(33.0, abs=1e-6)


def test_z_is_zero_before_lock():
    s = SkinTemperatureChannelState()
    s.observe_resting(skin_temp_c=33.0)
    assert skin_temperature_z(s, 30.0) == 0.0


def test_colder_skin_gives_positive_arousal_z():
    s = _locked_state(skin_c=33.0)
    # Skin DROPS below baseline -> vasoconstriction -> arousal -> z > 0.
    assert skin_temperature_z(s, 32.0) > 0.0
    # Skin warmer than baseline -> z < 0.
    assert skin_temperature_z(s, 34.0) < 0.0
    # At baseline -> ~0.
    assert skin_temperature_z(s, 33.0) == pytest.approx(0.0, abs=1e-6)


def test_evaluate_zero_confidence_before_lock():
    s = SkinTemperatureChannelState()
    s.observe_resting(skin_temp_c=33.0)
    out = evaluate_skin_temperature(s, skin_temp_c=31.0, ambient_temp_c=24.0)
    assert out.confidence == 0.0
    assert out.z == 0.0


def test_evaluate_zero_confidence_on_missing_sample():
    s = _locked_state()
    out = evaluate_skin_temperature(s, skin_temp_c=0.0, ambient_temp_c=24.0)
    assert out.confidence == 0.0
    assert out.z == 0.0


def test_confidence_full_when_ambient_stable():
    s = _locked_state(skin_c=33.0, ambient_c=24.0)
    out = evaluate_skin_temperature(s, skin_temp_c=32.0, ambient_temp_c=24.0)
    assert out.confidence == pytest.approx(1.0, abs=1e-6)
    assert out.z > 0.0


def test_ambient_drift_erodes_confidence():
    s = _locked_state(skin_c=33.0, ambient_c=24.0)
    # Ambient drifts halfway to the full-penalty band -> confidence ~0.5.
    half = TEMP_AMBIENT_DRIFT_C_FULL_PENALTY / 2.0
    out = evaluate_skin_temperature(
        s, skin_temp_c=32.0, ambient_temp_c=24.0 + half
    )
    assert out.confidence == pytest.approx(0.5, abs=1e-6)


def test_ambient_drift_past_band_zeroes_confidence():
    s = _locked_state(skin_c=33.0, ambient_c=24.0)
    out = evaluate_skin_temperature(
        s, skin_temp_c=32.0, ambient_temp_c=24.0 + TEMP_AMBIENT_DRIFT_C_FULL_PENALTY + 2.0
    )
    assert out.confidence == 0.0  # full drift -> channel contributes nothing


def test_missing_ambient_reference_does_not_penalise():
    # Baseline built with no ambient samples: cannot judge drift, so do not
    # punish the channel for missing ambient data.
    s = SkinTemperatureChannelState()
    for _ in range(TEMP_BASELINE_MIN_REST_EPOCHS):
        s.observe_resting(skin_temp_c=33.0)  # no ambient
    out = evaluate_skin_temperature(s, skin_temp_c=32.0, ambient_temp_c=24.0)
    assert out.confidence == pytest.approx(1.0, abs=1e-6)


def test_reset_clears_baseline():
    s = _locked_state()
    s.reset()
    assert not s.is_ready
    assert s.baseline_skin_c is None
    assert skin_temperature_z(s, 30.0) == 0.0
