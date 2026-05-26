"""
PPG peak detection (research/legacy): band-pass + systolic peak finding.

Used to *show what the IBI algorithm sees* — the filtered pulse wave with the
detected peaks overlaid — and to reconstruct IBI from PPG and measure pulse
amplitude (PPA). This is the kind of raw-PPG DSP the production path
deliberately does not run (wrist PPG is too motion-fragile for autonomic
features); it exists for the thesis demonstrations.

scipy is imported lazily so production stays free of it when the toggle is off.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from biofizic.config import (
    PPG_BAND_HI_HZ,
    PPG_BAND_LO_HZ,
    PPG_MIN_BEAT_DISTANCE_S,
    PPG_MIN_SAMPLES,
)


@dataclass(frozen=True)
class PpgPeakResult:
    peak_timestamps_ms: list[int] = field(default_factory=list)
    ppa: float = 0.0                 # pulse amplitude (filtered peak-to-trough)
    reconstructed_ibi_ms: list[int] = field(default_factory=list)
    sample_rate_hz: float = 0.0
    n_peaks: int = 0


def _effective_fs(timestamps_ms: list[int]) -> float:
    if len(timestamps_ms) < 2:
        return 0.0
    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    return (len(timestamps_ms) - 1) / span_s if span_s > 0 else 0.0


def detect_ppg_peaks(green: list[int], timestamps_ms: list[int]) -> PpgPeakResult:
    """Band-pass the green PPG and find systolic peaks. Returns peak timestamps,
    the median pulse amplitude, and the IBI reconstructed from peak spacing."""
    n = len(green)
    if n < PPG_MIN_SAMPLES or len(timestamps_ms) != n:
        return PpgPeakResult()

    fs = _effective_fs(timestamps_ms)
    nyq = fs / 2.0
    if fs <= 0 or PPG_BAND_HI_HZ >= nyq:
        return PpgPeakResult(sample_rate_hz=fs)

    # Lazy: scipy is a research-only dependency.
    from scipy.signal import butter, filtfilt, find_peaks

    sig = np.asarray(green, dtype=float)
    sig = sig - sig.mean()
    b, a = butter(2, [PPG_BAND_LO_HZ / nyq, PPG_BAND_HI_HZ / nyq], btype="band")
    # filtfilt needs more samples than the filter padding length.
    if n <= 3 * (max(len(a), len(b))):
        return PpgPeakResult(sample_rate_hz=fs)
    filt = filtfilt(b, a, sig)

    min_dist = max(1, int(PPG_MIN_BEAT_DISTANCE_S * fs))
    peaks, _ = find_peaks(filt, distance=min_dist)
    troughs, _ = find_peaks(-filt, distance=min_dist)

    ppa = 0.0
    if peaks.size and troughs.size:
        ppa = float(np.median(filt[peaks]) - np.median(filt[troughs]))

    peak_ts = [int(timestamps_ms[i]) for i in peaks]
    recon_ibi = [int(peak_ts[i + 1] - peak_ts[i]) for i in range(len(peak_ts) - 1)]

    return PpgPeakResult(
        peak_timestamps_ms=peak_ts,
        ppa=ppa,
        reconstructed_ibi_ms=recon_ibi,
        sample_rate_hz=fs,
        n_peaks=int(peaks.size),
    )
