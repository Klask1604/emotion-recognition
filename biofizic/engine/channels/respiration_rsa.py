"""
Respiration rate from RSA (respiratory sinus arrhythmia) in the IBI series.

Breathing modulates the heart rhythm: inhalation shortens IBIs, exhalation
lengthens them (RSA). The breathing frequency therefore shows up as a spectral
peak in the IBI series, classically in 0.15-0.4 Hz (9-24 breaths/min).

This estimator is deliberately honest about its own limits, because the target
user breathes slowly and quietly (< 10 br/min ≈ < 0.17 Hz): at low breathing
rates the RSA peak slides toward the LF band and blends with Mayer waves
(~0.1 Hz), so it cannot be cleanly separated. We therefore:

  - search a WIDENED band (RSA_BAND_LO_HZ..RSA_BAND_HI_HZ) so slow breathing is
    not missed outright, and
  - return a CONFIDENCE that collapses when the spectral peak is not clearly
    dominant or sits at the very bottom of the band (likely Mayer, not breath).

Uses Lomb-Scargle on the raw (unevenly sampled) IBI series rather than
resample+FFT: IBIs are samples at irregular beat times, and Lomb-Scargle is the
standard way to get a spectrum from unevenly sampled data without interpolation
artefacts.

Pure and side-effect free: feed it the recent IBI entries, get back an estimate.
NOT wired into the pipeline — this is the B2a isolated step, to be compared with
the PPG-amplitude estimator (B2b) before either is fused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biofizic.config import (
    RSA_BAND_HI_HZ,
    RSA_BAND_LO_HZ,
    RSA_MIN_BEATS,
    RSA_MIN_PEAK_PROMINENCE_RATIO,
    RSA_SLOW_EDGE_HZ,
)
from biofizic.ingestion.messages import InterbeatIntervalEntry


@dataclass(frozen=True)
class RespirationEstimate:
    """One respiration estimate plus an honest self-assessed confidence.

    breaths_per_min: peak frequency mapped to br/min, or 0 when unavailable.
    confidence:      [0, 1]; 0 when the series is too short, the peak is not
                     dominant, or it sits at the slow edge (Mayer-band ambiguity).
    peak_hz:         the detected peak frequency (diagnostic).
    prominence_ratio: peak power / median band power (diagnostic).
    """

    breaths_per_min: float
    confidence: float
    peak_hz: float
    prominence_ratio: float


def _beat_times_seconds(entries: list[InterbeatIntervalEntry]) -> np.ndarray:
    """Cumulative beat times (s) from IBI intervals; uses timestamps if present
    and coherent, else integrates the intervals."""
    ts = [e.timestamp_ms for e in entries]
    if all(t is not None for t in ts):
        t0 = ts[0]
        return np.array([(t - t0) / 1000.0 for t in ts], dtype=float)
    # Fall back to integrating intervals (place each beat at the running sum).
    times = np.cumsum([0.0] + [e.interval_ms / 1000.0 for e in entries[:-1]])
    return times.astype(float)


def estimate_respiration_rsa(entries: list[InterbeatIntervalEntry]) -> RespirationEstimate:
    """Estimate breathing rate from RSA via a Lomb-Scargle periodogram of the
    detrended IBI series. Returns confidence 0 (and rate 0) when no trustworthy
    peak can be found."""
    n = len(entries)
    if n < RSA_MIN_BEATS:
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    intervals = np.array([e.interval_ms for e in entries], dtype=float)
    if not np.all(np.isfinite(intervals)) or np.all(intervals == intervals[0]):
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    times = _beat_times_seconds(entries)
    duration = times[-1] - times[0]
    if duration <= 0:
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    # Detrend: RSA is the fluctuation around the mean IBI, not the mean itself.
    signal = intervals - float(np.mean(intervals))

    # Lomb-Scargle over the breathing band. scipy.signal.lombscargle wants
    # angular frequencies and a zero-mean signal.
    from scipy.signal import lombscargle

    freqs_hz = np.linspace(RSA_BAND_LO_HZ, RSA_BAND_HI_HZ, 200)
    angular = 2.0 * np.pi * freqs_hz
    try:
        power = lombscargle(times, signal, angular, normalize=True)
    except Exception:  # noqa: BLE001 — degenerate input
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    peak_idx = int(np.argmax(power))
    peak_hz = float(freqs_hz[peak_idx])
    peak_power = float(power[peak_idx])
    median_power = float(np.median(power)) or 1e-9
    prominence_ratio = peak_power / median_power

    # Confidence gate -----------------------------------------------------
    # 1) Peak must be clearly dominant over the band's typical power.
    if prominence_ratio < RSA_MIN_PEAK_PROMINENCE_RATIO:
        return RespirationEstimate(0.0, 0.0, peak_hz, prominence_ratio)
    # 2) A peak at the slow edge is more likely a Mayer wave than breathing;
    #    fade confidence as the peak approaches RSA_SLOW_EDGE_HZ.
    if peak_hz <= RSA_SLOW_EDGE_HZ:
        slow_factor = 0.0
    else:
        # Ramp from 0 at the slow edge to 1 a little above it.
        slow_factor = min(1.0, (peak_hz - RSA_SLOW_EDGE_HZ) / RSA_SLOW_EDGE_HZ)
    # 3) Prominence maps to a [0,1] confidence (saturating).
    prom_factor = min(1.0, (prominence_ratio - RSA_MIN_PEAK_PROMINENCE_RATIO)
                      / RSA_MIN_PEAK_PROMINENCE_RATIO)
    confidence = slow_factor * prom_factor

    breaths_per_min = peak_hz * 60.0
    return RespirationEstimate(
        breaths_per_min=breaths_per_min,
        confidence=confidence,
        peak_hz=peak_hz,
        prominence_ratio=prominence_ratio,
    )
