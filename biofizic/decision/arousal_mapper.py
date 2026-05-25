"""Map Kubios stress index to zones and arousal levels."""

from __future__ import annotations

from biofizic.config import (
    STRESS_INDEX_ZONE_BOUNDS,
    KubiosZone,
    KubiosZoneId,
    clip_value,
)


def kubios_zone_for_stress_index(stress_index: float) -> KubiosZone:
    """stress_index is sqrt(Baevsky SI) per Kubios HRV User Guide."""
    value = max(0.0, float(stress_index))
    for upper, zone_id, label, band_id, arousal_mid, arousal_10 in STRESS_INDEX_ZONE_BOUNDS:
        if value < upper:
            return KubiosZone(zone_id, label, band_id, arousal_mid, arousal_10)
    return KubiosZone(KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 0.95, 10)


def stress_index_to_arousal(stress_index: float) -> tuple[float, int, KubiosZone]:
    zone = kubios_zone_for_stress_index(stress_index)
    value = max(0.0, float(stress_index))
    lower_si = 0.0
    lower_arousal = 0.08
    for upper, *_rest, high_arousal, _a10 in STRESS_INDEX_ZONE_BOUNDS:
        if value < upper:
            span = max(0.01, upper - lower_si)
            fraction = (value - lower_si) / span
            arousal = clip_value(lower_arousal + fraction * (high_arousal - lower_arousal), 0.0, 1.0)
            scale_10 = int(round(clip_value(arousal * 10.0, 0.0, 10.0)))
            return arousal, scale_10, zone
        lower_si = upper
        lower_arousal = high_arousal
    return zone.arousal_mid, zone.arousal_scale_10, zone


def zone_is_alert_or_higher(zone: KubiosZone) -> bool:
    return zone.zone_id in (KubiosZoneId.HIGH, KubiosZoneId.VERY_HIGH)


def zone_is_elevated_or_higher(zone: KubiosZone) -> bool:
    return zone.zone_id in (
        KubiosZoneId.ELEVATED,
        KubiosZoneId.HIGH,
        KubiosZoneId.VERY_HIGH,
    )


def arousal_scale_10_to_label(scale_10: int) -> str:
    if scale_10 <= 2:
        return "Relaxat"
    if scale_10 <= 4:
        return "Echilibrat"
    if scale_10 <= 6:
        return "Moderat"
    if scale_10 <= 8:
        return "Alert"
    return "Ridicat"


def baseline_z_score_to_label(
    z_score: float,
    *,
    baseline_ready: bool,
    baseline_si: float = 0.0,
) -> str:
    """Map personal baseline z-score to Kubios zone label (same scale as arousal)."""
    if not baseline_ready or baseline_si <= 0:
        return "pending"
    scale = max(0.5, baseline_si * 0.15)
    effective_si = baseline_si + float(z_score) * scale
    return kubios_zone_for_stress_index(effective_si).label
