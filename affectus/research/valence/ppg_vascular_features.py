"""
PPG vascular-tone features for valence, based on a VALIDATED physiological
mechanism, not an invented proxy.

Autonomic valence responses drive peripheral vasomotor tone: unpleasant/fear
states cause vasoconstriction, some pleasant/anger states cause vasodilation
(Frontiers Physiol. 2021 PPG review; vascular-reactivity literature). This tone
is directly readable from the PPG pulse wave:

  perfusion_index = AC / DC      (pulse amplitude over baseline; the standard
                                  clinical vasoconstriction/vasodilation index)
  ac_amplitude    = systolic peak-to-trough of the band-passed pulse
  dc_level        = mean of the raw signal (baseline perfusion)
  rise_time       = systolic upstroke time (vascular stiffness/tone)
  pulse_width     = beat duration at half amplitude
  reflection_idx  = secondary (dicrotic) peak height / systolic peak
                    (arterial reflection, changes with vasomotor tone)

All averaged over the window. Research only; pure function.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VASC_FEATURE_NAMES = [
    "perfusion_index", "ac_amp", "dc_level",
    "rise_time", "pulse_width", "reflection_idx",
]


@dataclass(frozen=True)
class VascFeatures:
    values: dict
    valid: bool

    def as_dict(self) -> dict:
        return dict(self.values)


def extract_vascular_features(green: list[int], timestamps_ms: list[int]) -> VascFeatures:
    empty = VascFeatures({k: 0.0 for k in VASC_FEATURE_NAMES}, False)
    n = len(green)
    if n < 64 or len(timestamps_ms) != n:
        return empty
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return empty
    fs = (n - 1) / span_s

    from scipy.signal import butter, filtfilt, find_peaks

    raw = np.asarray(green, dtype=float)
    dc = float(np.mean(raw))                  # baseline perfusion (DC)
    nyq = fs / 2.0
    hi = min(8.0, nyq * 0.95)
    if hi <= 0.5 or fs <= 0:
        return empty
    b, a = butter(2, [0.5 / nyq, hi / nyq], btype="band")
    if n <= 3 * max(len(a), len(b)):
        return empty
    ac = filtfilt(b, a, raw - dc)             # pulsatile (AC) component

    dist = max(1, int(0.4 * fs))
    peaks, _ = find_peaks(ac, distance=dist)
    troughs, _ = find_peaks(-ac, distance=dist)
    if len(peaks) < 2 or len(troughs) < 2:
        return empty

    troughs = np.asarray(troughs)
    amps, rises, widths, refl = [], [], [], []
    for pk in peaks:
        before = troughs[troughs < pk]
        after = troughs[troughs > pk]
        if before.size == 0 or after.size == 0:
            continue
        t0, t1 = int(before[-1]), int(after[0])
        amp = float(ac[pk] - ac[t0])
        if amp <= 0:
            continue
        amps.append(amp)
        rises.append((pk - t0) / fs)
        # half-amplitude width
        half = ac[t0] + amp / 2.0
        seg = ac[t0:t1 + 1]
        above = np.where(seg >= half)[0]
        widths.append((above[-1] - above[0]) / fs if above.size > 1 else (t1 - t0) / fs)
        # reflection: a secondary local max between this peak and the next trough
        dseg = ac[pk:t1 + 1]
        sub, _ = find_peaks(dseg)
        refl.append(float(np.max(dseg[sub]) - ac[t1]) / amp if sub.size else 0.0)

    if not amps:
        return empty
    ac_amp = float(np.mean(amps))
    pi = ac_amp / abs(dc) if dc != 0 else 0.0
    return VascFeatures({
        "perfusion_index": pi,
        "ac_amp": ac_amp,
        "dc_level": dc,
        "rise_time": float(np.mean(rises)) if rises else 0.0,
        "pulse_width": float(np.mean(widths)) if widths else 0.0,
        "reflection_idx": float(np.mean(refl)) if refl else 0.0,
    }, True)
