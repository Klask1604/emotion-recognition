#!/usr/bin/env python3
"""
Biofizic context engine: activity_mode from acc_rms window.

Personal baseline: GMM on quiet epochs (mu, sigma) + Tukey-relative thresholds.
Classification: REST / SEDENTARY / ARM_ACTIVE / LOCOMOTION with hysteresis.

Methodology:
  - Baseline: AdaptiveMotionBaseline (GMM 1-comp quiet -> mu, sigma)
  - Thresholds: mu + k * sigma (Tukey inner fence, not fixed m/s^2)
  - Auxiliary: 2-comp GMM rest/active intersection per session
  - motion_z = (acc - mu) / sigma for Grafana / Unity
  - LOCOMOTION: veto when instant acc is low (stale p90 ignored)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Sequence

import numpy as np

from biofizic.adaptive_motion import AdaptiveMotionBaseline, motion_z as compute_motion_z
from biofizic.paths import motion_model_path as default_motion_model_path

SCHEMA_VERSION = 4

# ── Fereastră acc_rms ─────────────────────────────────────────────────────────
WINDOW_SAMPLES = 10
MIN_SAMPLES_CLASSIFY = 3
RECENT_SAMPLES = 3
INSTANT_QUIET_EXIT = 2

# Multiplicatori Tukey pe σ personal (σ din GMM quiet sau MAD)
TUKEY_K_REST = 1.5
TUKEY_K_SEDENTARY = 3.0
TUKEY_K_ARM = 4.5
TUKEY_K_LOCOMOTION = 6.0
LOCOMOTION_FRAC_HIGH = 0.30

MODE_ENTER_STREAK = 3
MODE_EXIT_STREAK = 3
MODE_EXIT_STREAK_LOCOMOTION = 2

QUIET_LEARN_MAX_MPS2 = 1.0
QUIET_SAMPLES_MIN = 12
MAD_FLOOR_MPS2 = 0.08

# Fallback populație până la baseline învățat
FALLBACK_REST_MEDIAN_MAX = 0.40
FALLBACK_REST_P90_MAX = 0.55
FALLBACK_SEDENTARY_MEDIAN_MAX = 1.05
FALLBACK_SEDENTARY_P90_MAX = 1.40
FALLBACK_ARM_P90_MIN = 1.05
FALLBACK_ARM_MEDIAN_MAX = 2.0
FALLBACK_LOCOMOTION_MEDIAN_MIN = 1.55
FALLBACK_LOCOMOTION_P90_MIN = 2.05
FALLBACK_LOCOMOTION_HIGH_MPS2 = 2.2

# LOCOMOTION necesită mișcare reală — evită fals pozitiv când GMM baseline e foarte mic (~0.15)
LOCOMOTION_MIN_INSTANT_MPS2 = 1.0
LOCOMOTION_MIN_INSTANT_INTERACTIVE_MPS2 = 1.8

DEFAULT_MOTION_MODEL = default_motion_model_path()


class ActivityMode(str, Enum):
    REST = "REST"
    SEDENTARY = "SEDENTARY"
    ARM_ACTIVE = "ARM_ACTIVE"
    LOCOMOTION = "LOCOMOTION"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class ActivityContext:
    """Rezultat per epocă — compatibil cu vechiul MotionState (`.active`)."""

    mode: ActivityMode
    confidence: float
    acc_rms: float
    acc_median: float
    acc_p90: float
    acc_max: float
    acc_std: float
    frac_above_1: float
    frac_above_high: float
    quiet_baseline: float | None
    quiet_mad: float | None
    window_n: int
    mode_epochs: int
    motion_z: float = 0.0
    motion_baseline_source: str = "none"
    gmm_active_threshold: float | None = None
    expect_hand_motion: bool = False

    @property
    def active(self) -> bool:
        return self.mode in (ActivityMode.LOCOMOTION, ActivityMode.ARM_ACTIVE)

    @property
    def is_rest_like(self) -> bool:
        return self.mode in (ActivityMode.REST, ActivityMode.SEDENTARY)

    @property
    def is_physical_exertion(self) -> bool:
        return self.mode in (ActivityMode.LOCOMOTION, ActivityMode.ARM_ACTIVE)

    @property
    def suppress_stress_alert(self) -> bool:
        if self.confidence < 0.55:
            return False
        if self.mode == ActivityMode.LOCOMOTION:
            return True
        if self.mode == ActivityMode.ARM_ACTIVE and not self.expect_hand_motion:
            return True
        return False

    def activity_legacy(self) -> str:
        if self.acc_rms <= 0:
            return "?"
        if self.mode == ActivityMode.LOCOMOTION:
            return "mers"
        if self.mode == ActivityMode.ARM_ACTIVE:
            return "miscare"
        return "sedere"


def _mad(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    arr = np.asarray(values, dtype=float)
    med = float(np.median(arr))
    return float(np.median(np.abs(arr - med)))


def _thresholds(
    quiet_baseline: float | None,
    quiet_mad: float | None,
) -> dict[str, float]:
    if quiet_baseline is not None and quiet_mad is not None and quiet_mad > 0:
        b, m = quiet_baseline, quiet_mad
        return {
            "rest_med": b + TUKEY_K_REST * m,
            "rest_p90": b + (TUKEY_K_REST + 1.0) * m,
            "sed_med": b + TUKEY_K_SEDENTARY * m,
            "sed_p90": b + (TUKEY_K_SEDENTARY + 1.0) * m,
            "arm_p90": b + TUKEY_K_ARM * m,
            "arm_med_max": b + (TUKEY_K_ARM + 2.0) * m,
            "loc_med": b + TUKEY_K_LOCOMOTION * m,
            "loc_p90": b + (TUKEY_K_LOCOMOTION + 1.0) * m,
            "loc_high": b + TUKEY_K_LOCOMOTION * m,
        }
    return {
        "rest_med": FALLBACK_REST_MEDIAN_MAX,
        "rest_p90": FALLBACK_REST_P90_MAX,
        "sed_med": FALLBACK_SEDENTARY_MEDIAN_MAX,
        "sed_p90": FALLBACK_SEDENTARY_P90_MAX,
        "arm_p90": FALLBACK_ARM_P90_MIN,
        "arm_med_max": FALLBACK_ARM_MEDIAN_MAX,
        "loc_med": FALLBACK_LOCOMOTION_MEDIAN_MIN,
        "loc_p90": FALLBACK_LOCOMOTION_P90_MIN,
        "loc_high": FALLBACK_LOCOMOTION_HIGH_MPS2,
    }


def _instant_rest_like(acc_instant: float, t: dict[str, float]) -> bool:
    return acc_instant > 0 and acc_instant <= t["sed_med"]


def _locomotion_supported(
    med: float,
    p90_recent: float,
    frac_hi: float,
    acc_instant: float,
    t: dict[str, float],
    *,
    gmm_active_threshold: float | None,
    interactive_mode: bool = False,
) -> bool:
    loc_floor = (
        LOCOMOTION_MIN_INSTANT_INTERACTIVE_MPS2
        if interactive_mode
        else LOCOMOTION_MIN_INSTANT_MPS2
    )
    if acc_instant < loc_floor:
        return False
    if acc_instant >= t["loc_med"] * 0.80:
        return True
    if gmm_active_threshold is not None and gmm_active_threshold >= loc_floor:
        if acc_instant >= max(gmm_active_threshold * 0.95, loc_floor):
            if med >= t["sed_med"] and p90_recent >= gmm_active_threshold * 0.85:
                return True
    if med >= t["loc_med"] and p90_recent >= t["loc_p90"] * 0.85:
        return True
    if frac_hi >= LOCOMOTION_FRAC_HIGH and p90_recent >= t["arm_p90"]:
        return True
    return False


def _arm_active_supported(
    p90_recent: float,
    med: float,
    acc_instant: float,
    t: dict[str, float],
    *,
    gmm_active_threshold: float | None,
) -> bool:
    if acc_instant >= t["arm_p90"] * 0.75:
        return True
    if (
        gmm_active_threshold is not None
        and acc_instant >= gmm_active_threshold * 0.70
        and med <= t["arm_med_max"]
    ):
        return True
    return p90_recent >= t["arm_p90"] and med <= t["arm_med_max"]


def _classify_raw(
    med: float,
    p90: float,
    p90_recent: float,
    frac_hi: float,
    acc_instant: float,
    *,
    quiet_baseline: float | None,
    quiet_mad: float | None,
    gmm_active_threshold: float | None,
    interactive_mode: bool = False,
) -> tuple[ActivityMode, float]:
    t = _thresholds(quiet_baseline, quiet_mad)
    p90_use = min(p90, p90_recent) if p90_recent > 0 else p90

    if _instant_rest_like(acc_instant, t) and med <= t["sed_med"]:
        if med <= t["rest_med"] and p90_use <= t["rest_p90"]:
            conf = 0.85 + min(0.10, (t["rest_med"] - med) * 0.2)
            return ActivityMode.REST, clip(conf, 0.0, 0.95)
        conf = 0.72 + min(0.18, (t["sed_med"] - med) * 0.12)
        return ActivityMode.SEDENTARY, clip(conf, 0.0, 0.90)

    if med <= t["rest_med"] and p90_use <= t["rest_p90"]:
        conf = 0.85 + min(0.10, (t["rest_med"] - med) * 0.2)
        return ActivityMode.REST, clip(conf, 0.0, 0.95)

    if _locomotion_supported(
        med, p90_recent, frac_hi, acc_instant, t,
        gmm_active_threshold=gmm_active_threshold,
        interactive_mode=interactive_mode,
    ):
        conf = 0.55
        if med >= t["loc_med"]:
            conf += 0.15
        if p90_recent >= t["loc_p90"]:
            conf += 0.15
        if frac_hi >= LOCOMOTION_FRAC_HIGH:
            conf += 0.15
        return ActivityMode.LOCOMOTION, clip(conf, 0.0, 0.95)

    if _arm_active_supported(
        p90_recent, med, acc_instant, t,
        gmm_active_threshold=gmm_active_threshold,
    ):
        conf = 0.50 + min(0.35, (max(p90_recent, acc_instant) - t["arm_p90"]) * 0.25)
        return ActivityMode.ARM_ACTIVE, clip(conf, 0.0, 0.90)

    if med <= t["sed_med"] and p90_use <= t["sed_p90"]:
        conf = 0.70 + min(0.20, (t["sed_med"] - med) * 0.15)
        return ActivityMode.SEDENTARY, clip(conf, 0.0, 0.90)

    if p90_recent >= t["arm_p90"] * 0.85 and not _instant_rest_like(acc_instant, t):
        return ActivityMode.ARM_ACTIVE, 0.45
    return ActivityMode.SEDENTARY, 0.50


def clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dampen_hr_z(z_hr: float, ctx: ActivityContext) -> float:
    if not ctx.is_physical_exertion or ctx.confidence < 0.45:
        return z_hr
    factor = 0.40 if ctx.mode == ActivityMode.ARM_ACTIVE else 0.25
    return z_hr * factor


def _resolve_motion_model_path(path: Path | None) -> Path | None:
    if path is not None:
        return path
    return DEFAULT_MOTION_MODEL


class ActivityContextEngine:
    """
    Fereastră acc_rms + baseline GMM adaptiv + mod stabil cu histerezis.

    Apel: ctx = engine.update(acc_rms, allow_quiet_learning=True)
    """

    def __init__(
        self,
        motion_model_path: Path | None = None,
        *,
        persist_motion_model: bool = True,
    ) -> None:
        self._window: deque[float] = deque(maxlen=WINDOW_SAMPLES)
        self._baseline = AdaptiveMotionBaseline(
            _resolve_motion_model_path(motion_model_path),
            persist=persist_motion_model,
        )
        self._mode = ActivityMode.UNKNOWN
        self._mode_epochs = 0
        self._candidate = ActivityMode.UNKNOWN
        self._candidate_streak = 0
        self._instant_quiet_streak = 0
        self._expect_hand_motion = False

    def set_expect_hand_motion(self, value: bool) -> None:
        self._expect_hand_motion = value

    def update(self, acc_rms: float, *, allow_quiet_learning: bool) -> ActivityContext:
        if acc_rms > 0:
            self._window.append(acc_rms)

        self._baseline.observe(acc_rms, allow_quiet=allow_quiet_learning)
        bl = self._baseline.current()
        quiet_baseline, quiet_mad = bl.as_quiet_pair()
        if quiet_baseline is not None and quiet_mad is not None:
            quiet_mad = max(MAD_FLOOR_MPS2, quiet_mad)
        elif len(self._window) >= QUIET_SAMPLES_MIN and allow_quiet_learning:
            # compatibilitate dacă buffer quiet gol dar fereastra e liniștită
            arr_q = np.asarray(
                [v for v in self._window if 0 < v <= QUIET_LEARN_MAX_MPS2],
                dtype=float,
            )
            if arr_q.size >= QUIET_SAMPLES_MIN:
                quiet_baseline = float(np.median(arr_q))
                quiet_mad = max(MAD_FLOOR_MPS2, _mad(arr_q) * 1.4826)

        mz = compute_motion_z(acc_rms, bl)
        n = len(self._window)

        if n < MIN_SAMPLES_CLASSIFY:
            return ActivityContext(
                mode=ActivityMode.UNKNOWN,
                confidence=0.0,
                acc_rms=acc_rms,
                acc_median=acc_rms if acc_rms > 0 else 0.0,
                acc_p90=acc_rms if acc_rms > 0 else 0.0,
                acc_max=acc_rms if acc_rms > 0 else 0.0,
                acc_std=0.0,
                frac_above_1=0.0,
                frac_above_high=0.0,
                quiet_baseline=quiet_baseline,
                quiet_mad=quiet_mad,
                window_n=n,
                mode_epochs=self._mode_epochs,
                motion_z=mz,
                motion_baseline_source=bl.source,
                gmm_active_threshold=bl.active_threshold,
                expect_hand_motion=self._expect_hand_motion,
            )

        arr = np.asarray(self._window, dtype=float)
        med = float(np.median(arr))
        p90 = float(np.percentile(arr, 90))
        tail = arr[-RECENT_SAMPLES:] if n >= RECENT_SAMPLES else arr
        p90_recent = float(np.percentile(tail, 90))
        mx = float(np.max(arr))
        std = float(np.std(arr)) if n > 1 else 0.0
        frac_1 = float(np.mean(arr >= 1.0))
        loc_high = _thresholds(quiet_baseline, quiet_mad)["loc_high"]
        frac_hi = float(np.mean(arr >= loc_high))
        acc_instant = acc_rms if acc_rms > 0 else float(arr[-1])

        t = _thresholds(quiet_baseline, quiet_mad)
        if _instant_rest_like(acc_instant, t):
            self._instant_quiet_streak += 1
        else:
            self._instant_quiet_streak = 0

        raw_mode, raw_conf = _classify_raw(
            med,
            p90,
            p90_recent,
            frac_hi,
            acc_instant,
            quiet_baseline=quiet_baseline,
            quiet_mad=quiet_mad,
            gmm_active_threshold=bl.active_threshold,
            interactive_mode=self._expect_hand_motion,
        )

        if (
            self._mode in (ActivityMode.LOCOMOTION, ActivityMode.ARM_ACTIVE)
            and self._instant_quiet_streak >= INSTANT_QUIET_EXIT
            and raw_mode in (ActivityMode.REST, ActivityMode.SEDENTARY)
        ):
            raw_mode = ActivityMode.SEDENTARY
            raw_conf = max(raw_conf, 0.75)

        mode, conf = self._apply_hysteresis(raw_mode, raw_conf, acc_instant=acc_instant)

        if mode == self._mode:
            self._mode_epochs += 1
        else:
            self._mode_epochs = 1
            self._mode = mode

        return ActivityContext(
            mode=mode,
            confidence=conf,
            acc_rms=acc_rms,
            acc_median=med,
            acc_p90=p90,
            acc_max=mx,
            acc_std=std,
            frac_above_1=frac_1,
            frac_above_high=frac_hi,
            quiet_baseline=quiet_baseline,
            quiet_mad=quiet_mad,
            window_n=n,
            mode_epochs=self._mode_epochs,
            motion_z=mz,
            motion_baseline_source=bl.source,
            gmm_active_threshold=bl.active_threshold,
            expect_hand_motion=self._expect_hand_motion,
        )

    def _apply_hysteresis(
        self,
        raw_mode: ActivityMode,
        raw_conf: float,
        *,
        acc_instant: float = 0.0,
    ) -> tuple[ActivityMode, float]:
        if self._mode == ActivityMode.UNKNOWN:
            return raw_mode, raw_conf * 0.85

        if raw_mode == self._mode:
            self._candidate = raw_mode
            self._candidate_streak = 0
            return self._mode, raw_conf

        if raw_mode == self._candidate:
            self._candidate_streak += 1
        else:
            self._candidate = raw_mode
            self._candidate_streak = 1

        exit_need = MODE_EXIT_STREAK
        if self._mode == ActivityMode.LOCOMOTION:
            exit_need = MODE_EXIT_STREAK_LOCOMOTION
        enter_need = MODE_ENTER_STREAK

        if self._mode_rank(raw_mode) > self._mode_rank(self._mode):
            if self._candidate_streak >= enter_need:
                return raw_mode, raw_conf * 0.90
            return self._mode, raw_conf * 0.70

        if self._instant_quiet_streak >= INSTANT_QUIET_EXIT and acc_instant > 0:
            return raw_mode, raw_conf * 0.88

        if self._candidate_streak >= exit_need:
            return raw_mode, raw_conf * 0.85
        return self._mode, raw_conf * 0.75

    @staticmethod
    def _mode_rank(mode: ActivityMode) -> int:
        order = {
            ActivityMode.UNKNOWN: 0,
            ActivityMode.REST: 1,
            ActivityMode.SEDENTARY: 2,
            ActivityMode.ARM_ACTIVE: 3,
            ActivityMode.LOCOMOTION: 4,
        }
        return order.get(mode, 0)


def context_to_mqtt(ctx: ActivityContext) -> dict:
    return {
        "context_schema": SCHEMA_VERSION,
        "activity_mode": ctx.mode.value,
        "activity_confidence": round(ctx.confidence, 3),
        "acc_window_median": round(ctx.acc_median, 3),
        "acc_window_p90": round(ctx.acc_p90, 3),
        "acc_window_n": ctx.window_n,
        "quiet_baseline": round(ctx.quiet_baseline, 3) if ctx.quiet_baseline else None,
        "quiet_mad": round(ctx.quiet_mad, 3) if ctx.quiet_mad else None,
        "motion_z": round(ctx.motion_z, 3),
        "motion_baseline_source": ctx.motion_baseline_source,
        "gmm_active_threshold": (
            round(ctx.gmm_active_threshold, 3) if ctx.gmm_active_threshold else None
        ),
        "context_suppress_alert": ctx.suppress_stress_alert,
        "context_rest_like": ctx.is_rest_like,
        "expect_hand_motion": ctx.expect_hand_motion,
    }
