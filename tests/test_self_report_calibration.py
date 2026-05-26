"""Self-report calibration: reported arousal anchors the baseline via probit."""

from __future__ import annotations

from pathlib import Path

import pytest

from biofizic.engine.arousal_mapper import normal_cdf, normal_ppf, personal_arousal_10
from biofizic.engine.baseline import RestBaselineStore


def test_probit_is_cdf_inverse():
    assert normal_ppf(0.5) == pytest.approx(0.0, abs=1e-3)
    for z in (-1.5, -0.4, 0.7, 2.0):
        assert normal_ppf(normal_cdf(z)) == pytest.approx(z, abs=1e-3)


def test_offset_places_baseline_at_reported_arousal():
    # At z=0 the arousal should reflect the self-report, not a fixed 5.
    calm_offset = normal_ppf(0.2)
    stressed_offset = normal_ppf(0.8)
    assert personal_arousal_10(0.0, calm_offset) < personal_arousal_10(0.0, stressed_offset)
    # Calm report -> low arousal at baseline; stressed -> high.
    assert personal_arousal_10(0.0, calm_offset) <= 3
    assert personal_arousal_10(0.0, stressed_offset) >= 8


def test_baseline_stores_and_applies_reported_arousal(tmp_path: Path):
    store = RestBaselineStore(path=tmp_path / "rest_baseline.json")
    store.reset_for_recalibration(reported_arousal=0.2)
    assert store.reported_baseline_arousal == pytest.approx(0.2)
    assert store.arousal_offset_z < 0  # calm => negative z-offset
    # Persists across reloads.
    reloaded = RestBaselineStore(path=tmp_path / "rest_baseline.json")
    assert reloaded.reported_baseline_arousal == pytest.approx(0.2)
