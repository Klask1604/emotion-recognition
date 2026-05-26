"""Kubios zone bounds, normal-CDF personal arousal, and label mapper."""

from __future__ import annotations

import pytest

from biofizic.config import KubiosZoneId
from biofizic.engine.arousal_mapper import (
    arousal_scale_10_to_label,
    cohen_kappa,
    kubios_zone_for_stress_index,
    normal_cdf,
    personal_arousal_10,
    population_arousal_10,
)


@pytest.mark.parametrize(
    "stress_index, expected_zone",
    [
        (0.0, KubiosZoneId.LOW),
        (7.0, KubiosZoneId.LOW),       # just below the 7.1 LOW boundary
        (7.2, KubiosZoneId.NORMAL),    # just above
        (12.1, KubiosZoneId.NORMAL),
        (12.3, KubiosZoneId.ELEVATED),
        (22.5, KubiosZoneId.HIGH),
        (29.9, KubiosZoneId.HIGH),
        (30.1, KubiosZoneId.VERY_HIGH),
        (100.0, KubiosZoneId.VERY_HIGH),
    ],
)
def test_kubios_zone_boundaries(stress_index: float, expected_zone: KubiosZoneId):
    assert kubios_zone_for_stress_index(stress_index).zone_id == expected_zone


def test_population_arousal_increases_with_zone():
    assert population_arousal_10(3.0) < population_arousal_10(10.0) < population_arousal_10(25.0)


def test_personal_arousal_centered_at_baseline():
    # z = 0 -> Phi = 0.5 -> arousal ~ 5.5 -> rounds to 5 or 6.
    assert normal_cdf(0.0) == pytest.approx(0.5)
    assert personal_arousal_10(0.0) in (5, 6)
    assert personal_arousal_10(-3.0) == 1
    assert personal_arousal_10(3.0) == 10
    # Monotonic in z.
    assert personal_arousal_10(-1.0) < personal_arousal_10(0.0) < personal_arousal_10(1.0)


def test_arousal_scale_10_to_label():
    assert arousal_scale_10_to_label(1) == "Relaxat"
    assert arousal_scale_10_to_label(3) == "Echilibrat"
    assert arousal_scale_10_to_label(5) == "Moderat"
    assert arousal_scale_10_to_label(7) == "Alert"
    assert arousal_scale_10_to_label(9) == "Ridicat"


def test_cohen_kappa_perfect_and_chance():
    labels = ["Relaxat", "Moderat", "Alert", "Relaxat"]
    assert cohen_kappa(labels, labels) == pytest.approx(1.0)
    # Identical marginals but never matching -> kappa < 0.
    assert cohen_kappa(["A", "B"], ["B", "A"]) < 0
