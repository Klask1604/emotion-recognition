"""
HRV (heart-rate variability) features from PPG-derived inter-beat intervals.

These are DIFFERENT from valence_ppg_fd (spectral shape of the raw pulse wave)
and valence_ppg_morph (pulse morphology): here we detect systolic peaks, build
the IBI series, and compute classic HRV indices that index vagal/sympathetic
balance, the autonomic axis linked (weakly) to affect in the literature.

  Time-domain:  mean_hr, sdnn, rmssd, pnn50
  Freq-domain:  lf_power (0.04-0.15 Hz), hf_power (0.15-0.40 Hz), lf_hf_ratio
                (computed via Lomb-Scargle on the unevenly-sampled IBI series)

Research only; pure function. Returns valid=False when too few beats.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

HRV_FEATURE_NAMES = [
    "hrv_mean_hr", "hrv_sdnn", "hrv_rmssd", "hrv_pnn50",
    "hrv_lf", "hrv_hf", "hrv_lf_hf",
]


@dataclass(frozen=True)
class HrvFeatures:
    values: dict
    valid: bool

    def as_dict(self) -> dict:
        return dict(self.values)


def _ibi_from_ppg(green: list[int], fs: float) -> np.ndarray:
    """Detect systolic peaks and return the IBI series in ms."""
    from scipy.signal import butter, filtfilt, find_peaks

    sig = np.asarray(green, dtype=float)
    sig = sig - sig.mean()
    nyq = fs / 2.0
    hi = min(4.0, nyq * 0.95)
    if hi <= 0.5 or fs <= 0:
        return np.array([])
    b, a = butter(2, [0.5 / nyq, hi / nyq], btype="band")
    if len(sig) <= 3 * max(len(a), len(b)):
        return np.array([])
    filt = filtfilt(b, a, sig)
    peaks, _ = find_peaks(filt, distance=int(0.4 * fs))
    if len(peaks) < 4:
        return np.array([])
    ibi = np.diff(peaks) / fs * 1000.0  # ms
    # physiological filter
    ibi = ibi[(ibi >= 300) & (ibi <= 2000)]
    return ibi


def extract_hrv_features(green: list[int], timestamps_ms: list[int]) -> HrvFeatures:
    empty = HrvFeatures({k: 0.0 for k in HRV_FEATURE_NAMES}, False)
    n = len(green)
    if n < 64 or len(timestamps_ms) != n:
        return empty
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    if span_s <= 0:
        return empty
    fs = (n - 1) / span_s
    ibi = _ibi_from_ppg(green, fs)
    if len(ibi) < 4:
        return empty

    mean_ibi = float(np.mean(ibi))
    mean_hr = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0
    sdnn = float(np.std(ibi, ddof=1)) if len(ibi) > 1 else 0.0
    diffs = np.diff(ibi)
    rmssd = float(np.sqrt(np.mean(diffs ** 2))) if len(diffs) else 0.0
    pnn50 = float(100.0 * np.sum(np.abs(diffs) > 50) / len(diffs)) if len(diffs) else 0.0

    # Frequency-domain via Lomb-Scargle on the IBI tachogram.
    lf = hf = 0.0
    if len(ibi) >= 8:
        t = np.cumsum(ibi) / 1000.0     # beat times (s)
        t = t - t[0]
        y = ibi - np.mean(ibi)
        try:
            from scipy.signal import lombscargle
            freqs = np.linspace(0.04, 0.40, 128)
            power = lombscargle(t, y, 2.0 * np.pi * freqs, normalize=True)
            lf = float(np.sum(power[(freqs >= 0.04) & (freqs < 0.15)]))
            hf = float(np.sum(power[(freqs >= 0.15) & (freqs <= 0.40)]))
        except Exception:  # noqa: BLE001
            lf = hf = 0.0
    lf_hf = lf / hf if hf > 0 else 0.0

    return HrvFeatures({
        "hrv_mean_hr": mean_hr,
        "hrv_sdnn": sdnn,
        "hrv_rmssd": rmssd,
        "hrv_pnn50": pnn50,
        "hrv_lf": lf,
        "hrv_hf": hf,
        "hrv_lf_hf": lf_hf,
    }, True)
