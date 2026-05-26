"""
Robust personal resting baseline (log-space).

RMSSD and the Kubios stress index are log-normally distributed (Task Force
ESC/NASPE 1996; Nunan 2010), so the baseline is estimated on ln(x) with a
robust location/scale: median and MAD, sigma = 1.4826*MAD (Hampel). The
personal z-score is z = (ln x - median) / sigma. This replaces the earlier
fixed 15%-of-baseline scale, which was not a statistic.

Rules:
  1. Collect resting epochs (quality-gated by the caller) into a rolling window.
  2. Lock after BASELINE_MIN_REST_EPOCHS; the estimate then slides over the
     last BASELINE_ROBUST_WINDOW_EPOCHS, adapting to circadian drift without an
     EMA fudge factor.
  3. Never reset on motion. Reset only via explicit recalibrate command.
  4. Persist to data/rest_baseline.json
"""

from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path

from biofizic.config import (
    BASELINE_LOG_SIGMA_FLOOR,
    BASELINE_MIN_REST_EPOCHS,
    BASELINE_ROBUST_WINDOW_EPOCHS,
    Z_SCORE_CLIP,
    rest_baseline_path,
)

log = logging.getLogger("rest_baseline")


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad_sigma(values: list[float], center: float) -> float:
    """Robust sigma estimate from the median absolute deviation (Hampel)."""
    mad = _median([abs(v - center) for v in values])
    return max(1.4826 * mad, BASELINE_LOG_SIGMA_FLOOR)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class RestBaselineStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or rest_baseline_path()
        self._ln_rmssd: deque[float] = deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_si: deque[float] = deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_hr: deque[float] = deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self.is_ready = False
        self.rest_observation_count = 0
        # Self-reported arousal at the last calibration (0..1). Sets where the
        # baseline sits on the arousal scale; 0.5 = neutral if never reported.
        self.reported_baseline_arousal = 0.5
        self._load()

    def observe_resting(
        self, rmssd_ms: float, kubios_stress_index: float, heart_rate_bpm: float = 0.0
    ) -> None:
        """Add a resting epoch. The caller decides what counts as resting
        (signal-quality gate: low motion energy and low artifact rate)."""
        if rmssd_ms <= 0 or kubios_stress_index <= 0:
            return
        self.rest_observation_count += 1
        self._ln_rmssd.append(math.log(rmssd_ms))
        self._ln_si.append(math.log(kubios_stress_index))
        if heart_rate_bpm > 0:
            self._ln_hr.append(math.log(heart_rate_bpm))
        if not self.is_ready and len(self._ln_si) >= BASELINE_MIN_REST_EPOCHS:
            self.is_ready = True
            log.info(
                "Baseline locked: stress_index=%.2f rmssd=%.1f (n=%d)",
                self.baseline_stress_index or 0.0,
                self.baseline_rmssd_ms or 0.0,
                len(self._ln_si),
            )
        self._save()

    def reset_for_recalibration(self, reported_arousal: float | None = None) -> None:
        """Only call from biofizic/cmd/calibrate. `reported_arousal` (0..1) is the
        user's self-reported state, anchoring where this baseline sits on the
        arousal scale."""
        self._ln_rmssd.clear()
        self._ln_si.clear()
        self._ln_hr.clear()
        self.is_ready = False
        self.rest_observation_count = 0
        if reported_arousal is not None:
            self.reported_baseline_arousal = min(max(float(reported_arousal), 0.0), 1.0)
        self._save()
        log.info(
            "Baseline reset for recalibration (reported_arousal=%.2f)",
            self.reported_baseline_arousal,
        )

    @property
    def arousal_offset_z(self) -> float:
        """Probit of the self-reported baseline arousal: the z-offset so that at
        z=0 the displayed arousal equals what the subject reported."""
        from biofizic.engine.arousal_mapper import normal_ppf

        return normal_ppf(self.reported_baseline_arousal)

    def stress_index_z_score(self, kubios_stress_index: float) -> float:
        """z>0 means elevated stress relative to the personal resting baseline."""
        if not self.is_ready or kubios_stress_index <= 0 or not self._ln_si:
            return 0.0
        values = list(self._ln_si)
        center = _median(values)
        sigma = _mad_sigma(values, center)
        z = (math.log(kubios_stress_index) - center) / sigma
        return _clip(z, Z_SCORE_CLIP)

    def rmssd_z_score(self, rmssd_ms: float) -> float:
        """z>0 means RMSSD below the personal resting baseline (i.e. stress)."""
        if not self.is_ready or rmssd_ms <= 0 or not self._ln_rmssd:
            return 0.0
        values = list(self._ln_rmssd)
        center = _median(values)
        sigma = _mad_sigma(values, center)
        z = (center - math.log(rmssd_ms)) / sigma
        return _clip(z, Z_SCORE_CLIP)

    def hr_z_score(self, heart_rate_bpm: float) -> float:
        """z>0 means HR above the personal resting baseline (arousal up).

        HR is robust to wrist motion (SDK-processed), so this anchors the
        fusion arousal when HRV is unreliable (exertion / motion)."""
        if not self.is_ready or heart_rate_bpm <= 0 or not self._ln_hr:
            return 0.0
        values = list(self._ln_hr)
        center = _median(values)
        sigma = _mad_sigma(values, center)
        z = (math.log(heart_rate_bpm) - center) / sigma
        return _clip(z, Z_SCORE_CLIP)

    @property
    def baseline_heart_rate_bpm(self) -> float | None:
        if not self._ln_hr:
            return None
        return math.exp(_median(list(self._ln_hr)))

    @property
    def baseline_stress_index(self) -> float | None:
        if not self._ln_si:
            return None
        return math.exp(_median(list(self._ln_si)))

    @property
    def baseline_rmssd_ms(self) -> float | None:
        if not self._ln_rmssd:
            return None
        return math.exp(_median(list(self._ln_rmssd)))

    def _save(self) -> None:
        payload = {
            "is_ready": self.is_ready,
            "rest_observation_count": self.rest_observation_count,
            "ln_si": list(self._ln_si),
            "ln_rmssd": list(self._ln_rmssd),
            "ln_hr": list(self._ln_hr),
            "reported_baseline_arousal": self.reported_baseline_arousal,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.is_ready = bool(data.get("is_ready", False))
            self.rest_observation_count = int(data.get("rest_observation_count", 0))
            self._ln_si.extend(float(v) for v in data.get("ln_si", []))
            self._ln_rmssd.extend(float(v) for v in data.get("ln_rmssd", []))
            self._ln_hr.extend(float(v) for v in data.get("ln_hr", []))
            self.reported_baseline_arousal = float(
                data.get("reported_baseline_arousal", 0.5)
            )
        except Exception as exc:
            log.warning("Could not load baseline file: %s", exc)
