"""
PPG morphological (pulse-shape) features for valence — the complement to the
frequency-domain features (valence_ppg_fd). Replicates the morphological feature
set from Frontiers Physiol. 2025 (PMC11893849, Table 4): per-pulse shape
descriptors of the rising and falling phases of each systolic wave.

The 14 features (averaged over all pulses in the window):
  Amp_Diff        peak-to-trough amplitude
  Interval_Rise   trough->peak duration (s)
  Interval_Drop   peak->next-trough duration (s)
  Slope_Rise      mean rising slope
  Slope_Max_Rise  max rising slope
  Slope_Drop      mean falling slope
  Slope_Min_Drop  min (steepest) falling slope
  Area_Rise       area under the rising phase
  Area_Drop       area under the falling phase
  Area_Total      total pulse area
  Area_Rise_rate  Area_Rise / Area_Total
  Area_Drop_rate  Area_Drop / Area_Total
  Area_Total_rate Area_Total normalised by pulse width
  Area_RD_rate    Area_Rise / Area_Drop

These capture autonomic effects on vascular tone (rise/fall asymmetry, dicrotic
shape) that the harmonic powers do not. Pure / side-effect free; research only.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# numpy 2.x renamed trapz -> trapezoid; support both.
_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))

from biofizic.config import (
    PPG_BAND_HI_HZ,
    PPG_BAND_LO_HZ,
    PPG_MIN_BEAT_DISTANCE_S,
)

MORPH_FEATURE_NAMES = [
    "amp_diff", "interval_rise", "interval_drop",
    "slope_rise", "slope_max_rise", "slope_drop", "slope_min_drop",
    "area_rise", "area_drop", "area_total",
    "area_rise_rate", "area_drop_rate", "area_total_rate", "area_rd_rate",
]


@dataclass(frozen=True)
class MorphFeatures:
    values: dict
    valid: bool

    def as_dict(self) -> dict:
        return dict(self.values)


def _bandpass(sig: np.ndarray, fs: float) -> np.ndarray | None:
    from scipy.signal import butter, filtfilt

    nyq = fs / 2.0
    hi = min(PPG_BAND_HI_HZ, nyq * 0.95)
    if hi <= PPG_BAND_LO_HZ or fs <= 0:
        return None
    centered = sig - float(np.mean(sig))
    b, a = butter(2, [PPG_BAND_LO_HZ / nyq, hi / nyq], btype="band")
    if len(centered) <= 3 * max(len(a), len(b)):
        return None
    return filtfilt(b, a, centered)


def extract_morph_features(green: list[int], timestamps_ms: list[int]) -> MorphFeatures:
    """Per-pulse shape features, averaged over the window."""
    empty = MorphFeatures({k: 0.0 for k in MORPH_FEATURE_NAMES}, False)
    n = len(green)
    if n < 64 or len(timestamps_ms) != n:
        return empty
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return empty
    fs = (n - 1) / span_s
    filt = _bandpass(np.asarray(green, dtype=float), fs)
    if filt is None:
        return empty

    from scipy.signal import find_peaks

    dist = max(1, int(PPG_MIN_BEAT_DISTANCE_S * fs))
    peaks, _ = find_peaks(filt, distance=dist)
    troughs, _ = find_peaks(-filt, distance=dist)
    if len(peaks) < 2 or len(troughs) < 2:
        return empty

    dt = 1.0 / fs
    feats = {k: [] for k in MORPH_FEATURE_NAMES}
    troughs = np.asarray(troughs)

    for pk in peaks:
        # nearest trough before (rise start) and after (drop end)
        before = troughs[troughs < pk]
        after = troughs[troughs > pk]
        if before.size == 0 or after.size == 0:
            continue
        t0, t1 = int(before[-1]), int(after[0])
        rise = filt[t0:pk + 1]
        drop = filt[pk:t1 + 1]
        if len(rise) < 2 or len(drop) < 2:
            continue

        amp = float(filt[pk] - filt[t0])
        if amp <= 0:
            continue
        interval_rise = (pk - t0) * dt
        interval_drop = (t1 - pk) * dt
        rise_slopes = np.diff(rise) / dt
        drop_slopes = np.diff(drop) / dt
        area_rise = float(_trapz(rise - filt[t0], dx=dt))
        area_drop = float(_trapz(drop - filt[t1], dx=dt))
        area_total = area_rise + area_drop
        width = interval_rise + interval_drop
        if area_total <= 0 or width <= 0:
            continue

        feats["amp_diff"].append(amp)
        feats["interval_rise"].append(interval_rise)
        feats["interval_drop"].append(interval_drop)
        feats["slope_rise"].append(float(np.mean(rise_slopes)))
        feats["slope_max_rise"].append(float(np.max(rise_slopes)))
        feats["slope_drop"].append(float(np.mean(drop_slopes)))
        feats["slope_min_drop"].append(float(np.min(drop_slopes)))
        feats["area_rise"].append(area_rise)
        feats["area_drop"].append(area_drop)
        feats["area_total"].append(area_total)
        feats["area_rise_rate"].append(area_rise / area_total)
        feats["area_drop_rate"].append(area_drop / area_total)
        feats["area_total_rate"].append(area_total / width)
        feats["area_rd_rate"].append(area_rise / area_drop if area_drop > 0 else 0.0)

    if not feats["amp_diff"]:
        return empty
    averaged = {k: float(np.mean(v)) if v else 0.0 for k, v in feats.items()}
    return MorphFeatures(averaged, True)
