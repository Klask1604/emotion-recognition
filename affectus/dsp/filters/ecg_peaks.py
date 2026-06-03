"""
ECG R-peak detection for calibration.

The Galaxy Watch Lead-I ECG (finger on the button) gives a clean QRS, far better
than wrist PPG for HRV. During calibration the server detects R-peaks here and
turns them into IBI to build the personal arousal baseline (the cleanest
reference possible). Reused machinery downstream: R-peak timestamps -> IBI ->
IbiBatchMessage -> the same RollingIbiBuffer/observe_resting path PPG uses.

Detection (a light Pan-Tompkins, not the full adaptive-threshold version — a 60 s
seated finger-hold needs no learning/search-back):
  - reject lead_off != 0 samples; detect only WITHIN contiguous lead-on segments
    so a finger lift never fabricates one giant IBI across the gap;
  - band-pass the QRS band, derivative + square (the energy step that keeps
    R-peaks dominant over T-waves), find_peaks with a 250 ms min spacing;
  - keep only IBIs in [ECG_IBI_MIN_MS, ECG_IBI_MAX_MS] so a missed/extra beat
    can't poison the baseline.

scipy is imported lazily (matches ppg_peaks.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from affectus.config import (
    ECG_BAND_HI_HZ,
    ECG_BAND_LO_HZ,
    ECG_IBI_MAX_MS,
    ECG_IBI_MIN_MS,
    ECG_MIN_BEAT_DISTANCE_S,
    ECG_MIN_SAMPLES,
)


@dataclass(frozen=True)
class EcgPeakResult:
    peak_timestamps_ms: list[int] = field(default_factory=list)
    reconstructed_ibi_ms: list[int] = field(default_factory=list)
    sample_rate_hz: float = 0.0
    n_peaks: int = 0
    lead_on_fraction: float = 0.0


def _effective_fs(timestamps_ms: list[int]) -> float:
    if len(timestamps_ms) < 2:
        return 0.0
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    return (len(timestamps_ms) - 1) / span_s if span_s > 0 else 0.0


def _lead_on_segments(
    lead_off: list[int] | None, n: int
) -> list[tuple[int, int]]:
    """Contiguous [start, end) index ranges where the lead is ON (lead_off == 0).
    When no lead_off is given, the whole window is one segment."""
    if lead_off is None:
        return [(0, n)]
    segments: list[tuple[int, int]] = []
    start: int | None = None
    for i in range(n):
        on = lead_off[i] == 0
        if on and start is None:
            start = i
        elif not on and start is not None:
            segments.append((start, i))
            start = None
    if start is not None:
        segments.append((start, n))
    return segments


def _rpeaks_in_segment(
    ecg: np.ndarray, ts: list[int], fs: float
) -> list[int]:
    """R-peak timestamps within one contiguous lead-on segment."""
    n = ecg.size
    nyq = fs / 2.0
    if fs <= 0 or ECG_BAND_HI_HZ >= nyq:
        return []
    from scipy.signal import butter, filtfilt, find_peaks

    sig = ecg - ecg.mean()
    b, a = butter(2, [ECG_BAND_LO_HZ / nyq, ECG_BAND_HI_HZ / nyq], btype="band")
    if n <= 3 * (max(len(a), len(b))):
        return []
    filt = filtfilt(b, a, sig)
    # Pan-Tompkins energy: derivative then square -> R-peaks dominate T-waves.
    energy = np.square(np.diff(filt, prepend=filt[0]))
    min_dist = max(1, int(ECG_MIN_BEAT_DISTANCE_S * fs))
    # Prominence relative to the segment's energy spread keeps low-amplitude
    # noise from registering as beats.
    prom = float(np.median(energy) + 0.5 * np.std(energy)) if energy.size else 0.0
    peaks, _ = find_peaks(energy, distance=min_dist, prominence=max(prom, 1e-9))
    return [int(ts[i]) for i in peaks]


def detect_ecg_rpeaks(
    ecg_mv: list[int | float],
    timestamps_ms: list[int],
    lead_off: list[int] | None = None,
) -> EcgPeakResult:
    """Detect R-peaks in a raw ECG window and reconstruct IBI from their spacing.

    Returns empty when the window is too short, the lead is mostly off, or the
    sample rate can't be inferred — the caller then simply doesn't feed the
    baseline this epoch (no fabricated beats)."""
    n = len(ecg_mv)
    if n < ECG_MIN_SAMPLES or len(timestamps_ms) != n:
        return EcgPeakResult()
    if lead_off is not None and len(lead_off) != n:
        lead_off = None  # malformed -> ignore, treat all as lead-on

    fs = _effective_fs(timestamps_ms)
    lead_on = (sum(1 for v in lead_off if v == 0) / n) if lead_off is not None else 1.0

    ecg = np.asarray(ecg_mv, dtype=float)
    peak_ts: list[int] = []
    for (s, e) in _lead_on_segments(lead_off, n):
        if e - s < ECG_MIN_SAMPLES:
            continue
        seg_fs = _effective_fs(timestamps_ms[s:e]) or fs
        peak_ts.extend(_rpeaks_in_segment(ecg[s:e], timestamps_ms[s:e], seg_fs))

    peak_ts.sort()
    # IBI from consecutive peaks, filtered to the physiological window. Peaks
    # across a lead-off gap are not consecutive within a segment, so no giant IBI.
    recon_ibi: list[int] = []
    for i in range(len(peak_ts) - 1):
        ibi = peak_ts[i + 1] - peak_ts[i]
        if ECG_IBI_MIN_MS <= ibi <= ECG_IBI_MAX_MS:
            recon_ibi.append(int(ibi))

    return EcgPeakResult(
        peak_timestamps_ms=peak_ts,
        reconstructed_ibi_ms=recon_ibi,
        sample_rate_hz=fs,
        n_peaks=len(peak_ts),
        lead_on_fraction=round(lead_on, 3),
    )
