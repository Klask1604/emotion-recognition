"""Map Kubios stress index to population zones and the personal z-score to arousal.

Two reference frames:
  - Kubios zones (population): SI breakpoints from the Kubios HRV User Guide
    (Baevsky 1984). Used for the population-referenced label and as the
    pre-baseline arousal fallback.
  - Personal arousal: once the per-subject baseline is locked, arousal is the
    normal CDF of the personal stress-index z-score, arousal = Phi(z), so the
    1..10 scale is anchored to the subject's own resting distribution instead
    of ad-hoc per-zone midpoints.
"""

from __future__ import annotations

import math

from biofizic.config import (
    STRESS_INDEX_ZONE_BOUNDS,
    KubiosZone,
    KubiosZoneId,
)


def kubios_zone_for_stress_index(stress_index: float) -> KubiosZone:
    """stress_index is sqrt(Baevsky SI) per Kubios HRV User Guide."""
    value = max(0.0, float(stress_index))
    for upper, zone_id, label, band_id, arousal_10 in STRESS_INDEX_ZONE_BOUNDS:
        if value < upper:
            return KubiosZone(zone_id, label, band_id, arousal_10)
    return KubiosZone(KubiosZoneId.VERY_HIGH, "Ridicat", "very_high", 10)


def population_arousal_10(stress_index: float) -> int:
    """Pre-baseline fallback: arousal level from the population Kubios zone."""
    return kubios_zone_for_stress_index(stress_index).arousal_scale_10


def normal_cdf(z: float) -> float:
    """Phi(z), the standard normal CDF, via erf (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(float(z) / math.sqrt(2.0)))


def normal_ppf(p: float) -> float:
    """Inverse normal CDF (probit), Acklam's rational approximation. Used to turn
    a self-reported arousal in (0,1) into a z-offset, so the baseline sits at the
    reported level instead of always at 0.5."""
    p = min(max(float(p), 1e-6), 1.0 - 1e-6)
    a = (-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00)
    b = (-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01)
    c = (-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00)
    d = (7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00)
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p <= phigh:
        q = p - 0.5
        r = q * q
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)
    q = math.sqrt(-2 * math.log(1 - p))
    return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)


def personal_arousal_10(z_score: float, offset_z: float = 0.0) -> int:
    """Arousal 1..10 from the personal z-score: round(1 + 9*Phi(z + offset)).

    offset_z is the probit of the self-reported baseline arousal, so at z=0 the
    arousal equals what the subject reported at calibration (not a fixed 5)."""
    phi = normal_cdf(z_score + offset_z)
    return int(round(1.0 + 9.0 * phi))


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


def cohen_kappa(labels_a: list[str], labels_b: list[str]) -> float:
    """Cohen's kappa agreement between two label sequences.

    Used to quantify how much the population-referenced (Kubios zone) and the
    subject-referenced (personal baseline) stress classifications agree over a
    session. kappa = (p_o - p_e) / (1 - p_e).
    """
    if len(labels_a) != len(labels_b) or not labels_a:
        return 0.0
    n = len(labels_a)
    categories = set(labels_a) | set(labels_b)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n
    expected = 0.0
    for c in categories:
        pa = labels_a.count(c) / n
        pb = labels_b.count(c) / n
        expected += pa * pb
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1.0 - expected)
