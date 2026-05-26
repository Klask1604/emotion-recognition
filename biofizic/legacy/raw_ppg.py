"""
Raw-PPG legacy engine: rolling pulse-wave buffer -> peaks, PPA, reconstructed IBI.

Demonstration only (published on biofizic/legacy/ppg). It also exposes a robust
PPA z-score (`ppa_z`) used by the valence heuristic, and the detected peak
timestamps for the "ALL DATA LIVE" dashboard overlay.
"""

from __future__ import annotations

from collections import deque

import numpy as np

from biofizic.config import PPG_ANALYSIS_WINDOW_S, PPG_PPA_BASELINE_WINDOW
from biofizic.dsp.ppg_peaks import detect_ppg_peaks

_PPA_SIGMA_FLOOR = 1.0  # linear PPA units; avoids div-by-zero when PPA is flat


class RawPpgEngine:
    def __init__(self) -> None:
        self._buf: deque[tuple[int, int]] = deque()
        self._ppa_hist: deque[float] = deque(maxlen=PPG_PPA_BASELINE_WINDOW)
        self.ppa_z: float = 0.0

    def process(self, batch) -> dict:
        for ts, g in zip(batch.ppg_timestamps_ms, batch.ppg_green):
            self._buf.append((int(ts), int(g)))
        if self._buf:
            cutoff = self._buf[-1][0] - int(PPG_ANALYSIS_WINDOW_S * 1000)
            while self._buf and self._buf[0][0] < cutoff:
                self._buf.popleft()

        timestamps = [t for t, _ in self._buf]
        green = [g for _, g in self._buf]
        res = detect_ppg_peaks(green, timestamps)

        if res.ppa > 0:
            self._ppa_hist.append(res.ppa)
            self.ppa_z = self._robust_z(res.ppa)

        recon_mean = (
            float(np.mean(res.reconstructed_ibi_ms)) if res.reconstructed_ibi_ms else 0.0
        )
        return {
            "n_peaks": res.n_peaks,
            "ppa": round(res.ppa, 2),
            "ppa_z": round(self.ppa_z, 2),
            "sample_rate_hz": round(res.sample_rate_hz, 1),
            "ibi_recon_mean": round(recon_mean, 1),
            "peak_ts": res.peak_timestamps_ms,
        }

    def _robust_z(self, x: float) -> float:
        vals = list(self._ppa_hist)
        if len(vals) < 5:
            return 0.0
        med = float(np.median(vals))
        mad = float(np.median(np.abs(np.array(vals) - med)))
        sigma = max(1.4826 * mad, _PPA_SIGMA_FLOOR)
        return float(np.clip((x - med) / sigma, -4.0, 4.0))
