"""Kubios zone bounds and the 1-10 arousal display label mapper."""

from __future__ import annotations

import pytest

from biofizic.config import KubiosZoneId
from biofizic.decision.arousal_mapper import (
    arousal_scale_10_to_label,
    kubios_zone_for_stress_index,
    stress_index_to_arousal,
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


def test_stress_index_to_arousal_monotonic_within_band():
    _, low, _ = stress_index_to_arousal(3.0)
    _, mid, _ = stress_index_to_arousal(10.0)
    _, hi, _ = stress_index_to_arousal(25.0)
    assert low <= mid <= hi


def test_arousal_scale_10_to_label():
    assert arousal_scale_10_to_label(1) == "Relaxat"
    assert arousal_scale_10_to_label(3) == "Echilibrat"
    assert arousal_scale_10_to_label(5) == "Moderat"
    assert arousal_scale_10_to_label(7) == "Alert"
    assert arousal_scale_10_to_label(9) == "Ridicat"
