"""
Peripheral skin-temperature arousal channel (isolated; not yet in the pipeline).

Physiology: sympathetic arousal triggers cutaneous vasoconstriction, so the
peripheral (wrist) skin temperature DROPS when arousal rises (Sensors 2026
multimodal SNS review; cutaneous vasoconstriction literature). Skin temperature
therefore complements HRV/HR by separating genuine arousal from movement, and is
already sampled by the watch (skin_temp / ambient_temp) but currently unused.

This module mirrors the design of engine/baseline.RestBaselineStore but in
LINEAR space (°C is not a positive multiplicative quantity like RMSSD/SI, so no
log transform): a robust personal baseline (median + MAD) of resting skin
temperature, with a z-score whose SIGN is inverted so that

    z > 0  ⇔  skin temperature is BELOW the personal resting baseline ⇔ arousal.

Quality gate: skin temperature is slow (minute-scale) and easily confounded by
ambient thermal drift and by simple wear-time warming. The channel reports a
confidence in [0, 1] that decays when the ambient temperature has drifted away
from its own resting baseline, because under exogenous thermal change a skin-
temperature drop is thermoregulation, not arousal. Confidence is also 0 until
the baseline has locked and 0 when no valid sample is present.

Everything here is pure and side-effect free except the explicit observe_* /
reset calls on the state object, so it can be unit-tested in isolation before
being wired into engine/decision.decide().
"""

from __future__ import annotations

import json
import logging
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

from biofizic.config import (
    BASELINE_OBSERVATION_INTERVAL_S,
    TEMP_AMBIENT_DRIFT_C_FULL_PENALTY,
    TEMP_BASELINE_MIN_REST_EPOCHS,
    TEMP_BASELINE_ROBUST_WINDOW_EPOCHS,
    TEMP_SIGMA_FLOOR_C,
    Z_SCORE_CLIP,
    temp_baseline_path,
)

log = logging.getLogger("temperature_channel")


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


def _mad_sigma(values: list[float], center: float, floor: float) -> float:
    """Robust sigma from the median absolute deviation (Hampel), with a floor so
    z stays finite when a subject's resting temperature is very stable."""
    mad = _median([abs(v - center) for v in values])
    return max(1.4826 * mad, floor)


def _clip(value: float, bound: float) -> float:
    return max(-bound, min(bound, value))


class SkinTemperatureChannelState:
    """Per-subject mutable state: rolling robust baselines for skin temperature
    and ambient temperature, plus the lock flag.

    Optional on-disk persistence (temp_baseline_path) so the channel survives a
    compute-engine restart, kept in a SEPARATE file from rest_baseline.json so
    adding it can never corrupt the HRV baseline. Passing path=None (the default
    in tests) keeps it purely in-memory.

    observe_resting takes the same optional `now` spacing gate as the HRV
    baseline: the temperature stream is slow and oversampled at ~1 Hz, so spaced
    samples keep the MAD meaningful and the lock honest."""

    def __init__(self, path: Path | None = None, *, persist: bool = False) -> None:
        self._path = path or (temp_baseline_path() if persist else None)
        self._skin_c: deque[float] = deque(maxlen=TEMP_BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ambient_c: deque[float] = deque(maxlen=TEMP_BASELINE_ROBUST_WINDOW_EPOCHS)
        self.is_ready = False
        self.rest_observation_count = 0
        self._last_observation_s: float | None = None
        if self._path is not None:
            self._load()

    def observe_resting(
        self,
        skin_temp_c: float,
        ambient_temp_c: float = 0.0,
        *,
        now: float | None = None,
    ) -> None:
        """Add a resting-epoch temperature sample. The caller decides what counts
        as resting (the same signal-quality gate that feeds the HRV baseline).
        Samples closer than BASELINE_OBSERVATION_INTERVAL_S are skipped when
        `now` is given (decorrelation); omit `now` to disable the gate."""
        if skin_temp_c <= 0:
            return
        if now is not None and self._last_observation_s is not None:
            if now - self._last_observation_s < BASELINE_OBSERVATION_INTERVAL_S:
                return
        if now is not None:
            self._last_observation_s = now
        self.rest_observation_count += 1
        self._skin_c.append(float(skin_temp_c))
        if ambient_temp_c > 0:
            self._ambient_c.append(float(ambient_temp_c))
        if not self.is_ready and len(self._skin_c) >= TEMP_BASELINE_MIN_REST_EPOCHS:
            self.is_ready = True
        self._save()

    def reset(self) -> None:
        self._skin_c.clear()
        self._ambient_c.clear()
        self.is_ready = False
        self.rest_observation_count = 0
        self._last_observation_s = None
        self._save()

    @property
    def baseline_skin_c(self) -> float | None:
        return _median(list(self._skin_c)) if self._skin_c else None

    @property
    def baseline_ambient_c(self) -> float | None:
        return _median(list(self._ambient_c)) if self._ambient_c else None

    def _save(self) -> None:
        if self._path is None:
            return
        payload = {
            "is_ready": self.is_ready,
            "rest_observation_count": self.rest_observation_count,
            "skin_c": list(self._skin_c),
            "ambient_c": list(self._ambient_c),
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
            self._skin_c.extend(float(v) for v in data.get("skin_c", []))
            self._ambient_c.extend(float(v) for v in data.get("ambient_c", []))
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not load temperature baseline: %s", exc)


@dataclass(frozen=True)
class SkinTemperatureArousal:
    """One epoch's temperature-channel output, shaped to drop straight into the
    decision fusion as a FusionChannel(z=z, weight=confidence)."""

    z: float            # >0 = skin colder than resting baseline = arousal
    confidence: float   # [0, 1]; 0 until baseline locks / under ambient drift
    skin_temp_c: float
    baseline_skin_c: float
    ambient_drift_c: float


def skin_temperature_z(state: SkinTemperatureChannelState, skin_temp_c: float) -> float:
    """Sign-inverted personal z: z>0 when skin temperature is below the resting
    baseline (sympathetic vasoconstriction). 0 until the baseline locks."""
    if not state.is_ready or skin_temp_c <= 0 or not state._skin_c:
        return 0.0
    values = list(state._skin_c)
    center = _median(values)
    sigma = _mad_sigma(values, center, TEMP_SIGMA_FLOOR_C)
    z = (center - float(skin_temp_c)) / sigma  # inverted: colder => higher z
    return _clip(z, Z_SCORE_CLIP)


def _ambient_drift_penalty(state: SkinTemperatureChannelState, ambient_temp_c: float) -> tuple[float, float]:
    """Return (confidence_multiplier, abs_drift_c). When the ambient temperature
    has drifted from its resting baseline, a skin-temperature change is more
    likely thermoregulation than arousal, so the channel is less trustworthy.
    Linear decay from 1.0 (no drift) to 0.0 at TEMP_AMBIENT_DRIFT_C_FULL_PENALTY."""
    baseline_ambient = state.baseline_ambient_c
    if baseline_ambient is None or ambient_temp_c <= 0:
        return 1.0, 0.0  # no ambient reference: do not penalise on missing data
    drift = abs(float(ambient_temp_c) - baseline_ambient)
    mult = max(0.0, 1.0 - drift / TEMP_AMBIENT_DRIFT_C_FULL_PENALTY)
    return mult, drift


def evaluate_skin_temperature(
    state: SkinTemperatureChannelState,
    *,
    skin_temp_c: float,
    ambient_temp_c: float = 0.0,
) -> SkinTemperatureArousal:
    """Score one epoch. confidence = baseline_ready · valid_sample · ambient_gate.

    The returned (z, confidence) plug directly into the fusion as a channel with
    weight = confidence, so a cold-start / drifting / missing-sample epoch
    contributes nothing (weight 0) — the same no-op guarantee the fusion relies
    on for unwired channels."""
    baseline_skin = state.baseline_skin_c or 0.0
    if not state.is_ready or skin_temp_c <= 0:
        return SkinTemperatureArousal(
            z=0.0, confidence=0.0, skin_temp_c=float(skin_temp_c),
            baseline_skin_c=baseline_skin, ambient_drift_c=0.0,
        )
    z = skin_temperature_z(state, skin_temp_c)
    ambient_mult, drift = _ambient_drift_penalty(state, ambient_temp_c)
    confidence = ambient_mult  # base trust is 1 once locked; ambient only erodes it
    return SkinTemperatureArousal(
        z=z, confidence=confidence, skin_temp_c=float(skin_temp_c),
        baseline_skin_c=baseline_skin, ambient_drift_c=drift,
    )
