"""
Personal valence baseline, recenters the cross-device WESAD model on the
subject.

The WESAD valence model generalises across subjects (LOSO ~82%) but, applied to
a different optical sensor (Galaxy Watch vs Empatica E4), carries a residual
domain-shift bias: a subject's resting pulse shape may resemble WESAD's "stress"
class, pinning the raw valence_z to one sign regardless of true affect. This is
the standard wearable problem and the standard fix: a generalist model plus
light per-subject calibration (Picard; DEAP/WESAD normalise valence per subject).

So the NEUTRAL point of valence is not assumed to be 0, it is MEASURED as the
subject's resting median of valence_z, and the SCALE is the resting MAD. The
personal valence is then  v = (valence_z - median) / sigma : >0 means more
positive than the subject's own neutral, <0 more negative. This mirrors
RestBaselineStore (HRV) and SkinTemperatureChannelState (temp), both of which
already estimate a robust personal baseline; valence is linear like temperature
(no log transform).

Persisted in its own file so it never touches the HRV baseline.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

from affectus.config import (
    BASELINE_MIN_REST_SAMPLES,
    BASELINE_OBSERVATION_INTERVAL_S,
    BASELINE_ROBUST_WINDOW_EPOCHS,
    VALENCE_SIGMA_FLOOR,
    Z_SCORE_CLIP,
    valence_baseline_path,
)

log = logging.getLogger("valence_baseline")


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad_sigma(values: list[float], center: float) -> float:
    """Robust sigma from the median absolute deviation (Hampel), floored so the
    personal z stays finite when resting valence is very stable. The floor only
    prevents divide-by-near-zero; the real scale is the measured MAD."""
    mad = _median([abs(v - center) for v in values])
    return max(1.4826 * mad, VALENCE_SIGMA_FLOOR)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class ValenceSmoother:
    """Rolling-mean smoother over a fixed time window (default 30 s).

    A single valence epoch (~1 s of PPG) is statistically too thin to assert an
    emotion: at rest the personal valence already varies ~1 sigma from
    physiological noise, and during activity it swings with real but brief events
    (a kill/death in a game). Averaging over WINDOW_S reports the BACKGROUND
    affective state (mood) rather than each second, the same reason arousal is
    smoothed over a 30-60 s HRV window, not published per second. This is signal
    aggregation, not hiding noise: the window matches the timescale on which an
    emotional state (not an event) actually persists."""

    def __init__(self, window_s: float = 30.0) -> None:
        self.window_s = window_s
        self._samples: deque[tuple[float, float]] = deque()  # (t_seconds, value)

    def update(self, value: float, now: float) -> float:
        """Add a sample and return the mean over the trailing window."""
        self._samples.append((now, float(value)))
        cutoff = now - self.window_s
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
        return sum(v for _, v in self._samples) / len(self._samples)

    def reset(self) -> None:
        self._samples.clear()


class ValenceBaselineStore:
    """Per-subject resting valence baseline (linear valence_z, median + MAD).

    Collects resting-epoch valence_z samples (the caller gates on the same
    signal-quality + still condition as the HRV baseline), locks after
    BASELINE_MIN_REST_SAMPLES spaced samples, and slides over the last
    BASELINE_ROBUST_WINDOW_EPOCHS so it adapts without an EMA fudge. path=None
    keeps it purely in-memory (tests)."""

    def __init__(self, path: Path | None = None, *, persist: bool = False) -> None:
        self._path = path or (valence_baseline_path() if persist else None)
        self._vz: deque[float] = deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self.is_ready = False
        self.rest_observation_count = 0
        self._last_observation_s: float | None = None
        # Self-reported resting valence (-1..+1) from the last calibration, and a
        # polarity flag set when the model's resting valence contradicts the
        # subject's report (cross-device sign inversion). Sign-only guard, it
        # never shifts the neutral, which stays the MEASURED resting median.
        self.reported_baseline_valence: float | None = None
        self._polarity_inverted = False
        if self._path is not None:
            self._load()

    def observe_resting(self, valence_z: float, *, now: float | None = None) -> None:
        """Add a resting valence_z sample. Spacing-gated like the HRV baseline so
        consecutive ~1 Hz samples (which share most of their PPG window) don't
        collapse the MAD."""
        if valence_z is None:
            return
        if now is not None and self._last_observation_s is not None:
            if now - self._last_observation_s < BASELINE_OBSERVATION_INTERVAL_S:
                return
        if now is not None:
            self._last_observation_s = now
        self.rest_observation_count += 1
        self._vz.append(float(valence_z))
        if not self.is_ready and len(self._vz) >= BASELINE_MIN_REST_SAMPLES:
            self.is_ready = True
            self._check_polarity()
            log.info("Valence baseline locked: neutral=%.3f (n=%d, polarity_inverted=%s)",
                     self.neutral_valence_z or 0.0, len(self._vz), self._polarity_inverted)
        self._save()

    def _check_polarity(self) -> None:
        """Sign-only cross-device guard. If the subject self-reported a clearly
        non-neutral resting valence (e.g. "pleasant") but the model's resting
        valence_z sits strongly on the OPPOSITE side of 0, the model's polarity is
        likely inverted on this subject, flag it so valence_personal flips sign.
        Coarse by design (a single resting report), and it never moves the neutral
        (which stays the measured median)."""
        self._polarity_inverted = False
        rv = self.reported_baseline_valence
        if rv is None or abs(rv) < 0.3:
            return  # no clear self-reported direction to check against
        median = self.neutral_valence_z
        if median is None or abs(median) < 0.3:
            return  # model's resting valence is near-neutral -> no contradiction
        if (rv > 0) != (median > 0):
            self._polarity_inverted = True
            log.warning("Valence polarity inverted: reported=%.2f vs model median=%.2f",
                        rv, median)

    def reset(self) -> None:
        self._vz.clear()
        self.is_ready = False
        self.rest_observation_count = 0
        self._last_observation_s = None
        self._save()

    def reset_for_recalibration(self, reported_valence: float | None = None) -> None:
        """Recalibrate: clear the resting samples and store the subject's
        self-reported resting valence (-1..+1) for the polarity sign-guard. The
        neutral itself is re-measured from the new resting epochs."""
        self._vz.clear()
        self.is_ready = False
        self.rest_observation_count = 0
        self._last_observation_s = None
        self._polarity_inverted = False
        if reported_valence is not None:
            self.reported_baseline_valence = max(-1.0, min(1.0, float(reported_valence)))
        self._save()
        log.info("Valence baseline reset for recalibration (reported_valence=%s)",
                 self.reported_baseline_valence)

    @property
    def neutral_valence_z(self) -> float | None:
        """The subject's measured resting median, the personal neutral point."""
        return _median(list(self._vz)) if self._vz else None

    def deadband(self, smoothing_factor: float = 0.45, reactivity_scale: float = 1.0) -> float:
        """Neutral-zone half-width in PERSONAL-z units, stable across restarts.

        valence_personal divides by the resting sigma, so a single resting epoch
        has spread ~1 in personal-z units. But the published value is SMOOTHED
        over 60 s, which shrinks the spread of independent-ish epochs by roughly
        sqrt(N_eff); empirically the smoothed resting spread is ~`smoothing_factor`
        of the raw. The dead-band is half of that smoothed resting spread, scaled
        by the subject's reactivity (low-responder -> narrower):

            deadband = 0.5 * smoothing_factor * reactivity_scale   (~0.22 at 1.0)

        It is a fixed multiple of the (constant, by construction) resting-z spread,
        so it does NOT drift between restarts the way an in-memory smoother window
        did."""
        return round(0.5 * smoothing_factor * reactivity_scale, 4)

    def valence_personal(self, valence_z: float) -> float:
        """Subject-centred valence: (valence_z - personal median) / personal MAD.
        0 until the baseline locks, so a pre-calibration verdict never claims a
        valence the calibration hasn't anchored. Sign is flipped when the polarity
        guard found the model inverted on this subject."""
        if not self.is_ready or not self._vz:
            return 0.0
        values = list(self._vz)
        center = _median(values)
        sigma = _mad_sigma(values, center)
        v = _clip((float(valence_z) - center) / sigma, Z_SCORE_CLIP)
        return -v if self._polarity_inverted else v

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "is_ready": self.is_ready,
            "rest_observation_count": self.rest_observation_count,
            "vz": list(self._vz),
            "reported_baseline_valence": self.reported_baseline_valence,
            "polarity_inverted": self._polarity_inverted,
        }
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def _load(self) -> None:
        if self._path is None or not self._path.exists():
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
            self.is_ready = bool(data.get("is_ready", False))
            self.rest_observation_count = int(data.get("rest_observation_count", 0))
            self._vz.extend(float(v) for v in data.get("vz", []))
            rbv = data.get("reported_baseline_valence")
            self.reported_baseline_valence = float(rbv) if rbv is not None else None
            self._polarity_inverted = bool(data.get("polarity_inverted", False))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load valence baseline: %s", exc)
