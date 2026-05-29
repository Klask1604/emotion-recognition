"""
Respiration rate from PPG pulse-amplitude modulation (B2b).

Breathing modulates not just the rhythm (RSA, see respiration_rsa) but also the
pulse AMPLITUDE: intra-thoracic pressure changes during the breathing cycle vary
the stroke volume / venous return, so the systolic peak-to-trough amplitude of
the PPG waveform rises and falls at the breathing frequency. This amplitude
modulation (often called RIAV) is generally MORE robust at slow breathing rates
than RSA, because it does not depend on vagal rhythm modulation that weakens with
age and slow/shallow breathing — which is exactly the target user's regime.

Pipeline:
  raw green PPG --(band-pass + peak detect, reuse dsp.ppg_peaks)--> per-beat
  systolic amplitudes -> a slowly-varying amplitude envelope sampled at the beat
  rate -> Lomb-Scargle over the breathing band -> dominant peak = breathing rate.

Same honest confidence gate as the RSA estimator: collapses when the amplitude
series is too short, the peak is not dominant, or it sits at the slow edge.

Pure / side-effect free; NOT wired into the pipeline. Lives beside the RSA
estimator so B2c can compare the two on real recordings before either is fused.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from biofizic.config import (
    RSA_BAND_HI_HZ,
    RSA_BAND_LO_HZ,
    RSA_MIN_PEAK_PROMINENCE_RATIO,
    RSA_SLOW_EDGE_HZ,
    RESP_PPG_MIN_BEATS,
)
from biofizic.engine.channels.respiration_rsa import RespirationEstimate


def _per_beat_amplitudes(
    green: list[int], timestamps_ms: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Return (beat_times_s, amplitude_series) from the PPG.

    Reuses the production peak detector, then takes the band-passed signal value
    at each detected peak minus the local trough as the per-beat amplitude. The
    resulting series, sampled at the (uneven) peak times, carries the breathing
    amplitude modulation."""
    n = len(green)
    if n < 1 or len(timestamps_ms) != n:
        return np.array([]), np.array([])

    span_s = (timestamps_ms[-1] - timestamps_ms[0]) / 1000.0
    fs = (n - 1) / span_s if span_s > 0 else 0.0
    nyq = fs / 2.0
    from biofizic.config import PPG_BAND_HI_HZ, PPG_BAND_LO_HZ, PPG_MIN_BEAT_DISTANCE_S

    if fs <= 0 or PPG_BAND_HI_HZ >= nyq:
        return np.array([]), np.array([])

    from scipy.signal import butter, filtfilt, find_peaks

    sig = np.asarray(green, dtype=float)
    sig = sig - sig.mean()
    b, a = butter(2, [PPG_BAND_LO_HZ / nyq, PPG_BAND_HI_HZ / nyq], btype="band")
    if n <= 3 * max(len(a), len(b)):
        return np.array([]), np.array([])
    filt = filtfilt(b, a, sig)

    min_dist = max(1, int(PPG_MIN_BEAT_DISTANCE_S * fs))
    peaks, _ = find_peaks(filt, distance=min_dist)
    if peaks.size < 2:
        return np.array([]), np.array([])

    ts = np.asarray(timestamps_ms, dtype=float)
    # Per-beat amplitude: peak value minus the preceding local minimum.
    amps = []
    beat_times = []
    for i, pk in enumerate(peaks):
        lo = peaks[i - 1] if i > 0 else 0
        trough_val = float(np.min(filt[lo:pk + 1]))
        amps.append(float(filt[pk]) - trough_val)
        beat_times.append((ts[pk] - ts[0]) / 1000.0)
    return np.asarray(beat_times), np.asarray(amps)


def estimate_respiration_ppg(
    green: list[int], timestamps_ms: list[int]
) -> RespirationEstimate:
    """Estimate breathing rate from PPG pulse-amplitude modulation. Returns
    confidence 0 (rate 0) when no trustworthy peak is found."""
    beat_times, amps = _per_beat_amplitudes(green, timestamps_ms)
    if amps.size < RESP_PPG_MIN_BEATS:
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)
    if not np.all(np.isfinite(amps)) or np.allclose(amps, amps[0]):
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    signal = amps - float(np.mean(amps))
    duration = beat_times[-1] - beat_times[0]
    if duration <= 0:
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    from scipy.signal import lombscargle

    freqs_hz = np.linspace(RSA_BAND_LO_HZ, RSA_BAND_HI_HZ, 200)
    angular = 2.0 * np.pi * freqs_hz
    try:
        power = lombscargle(beat_times, signal, angular, normalize=True)
    except Exception:  # noqa: BLE001
        return RespirationEstimate(0.0, 0.0, 0.0, 0.0)

    peak_idx = int(np.argmax(power))
    peak_hz = float(freqs_hz[peak_idx])
    peak_power = float(power[peak_idx])
    median_power = float(np.median(power)) or 1e-9
    prominence_ratio = peak_power / median_power

    if prominence_ratio < RSA_MIN_PEAK_PROMINENCE_RATIO:
        return RespirationEstimate(0.0, 0.0, peak_hz, prominence_ratio)
    if peak_hz <= RSA_SLOW_EDGE_HZ:
        slow_factor = 0.0
    else:
        slow_factor = min(1.0, (peak_hz - RSA_SLOW_EDGE_HZ) / RSA_SLOW_EDGE_HZ)
    prom_factor = min(1.0, (prominence_ratio - RSA_MIN_PEAK_PROMINENCE_RATIO)
                      / RSA_MIN_PEAK_PROMINENCE_RATIO)
    confidence = slow_factor * prom_factor

    return RespirationEstimate(
        breaths_per_min=peak_hz * 60.0,
        confidence=confidence,
        peak_hz=peak_hz,
        prominence_ratio=prominence_ratio,
    )
