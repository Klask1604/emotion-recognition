#!/usr/bin/env python3
"""
Baseline personal dual-EMA (rapid + lent) pentru HR și RMSSD.

- EMA rapid (~10 min): starea din sesiunea curentă
- EMA lent (~7 zile): normalul personal pe termen lung

Actualizare doar în activity_mode REST / SEDENTARY (context_engine).
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

TAU_FAST_SEC = 600.0       # ~10 min
TAU_SLOW_SEC = 604_800.0   # 7 zile
MIN_SLOW_SAMPLES = 6


def _ema_step(current: float | None, target: float, dt_sec: float, tau_sec: float) -> float:
    if current is None:
        return target
    dt = max(0.1, dt_sec)
    alpha = 1.0 - math.exp(-dt / tau_sec)
    return current + alpha * (target - current)


@dataclass
class BaselineSnapshot:
    hr_fast: float | None
    hr_slow: float | None
    rmssd_fast: float | None
    rmssd_slow: float | None
    slow_ready: bool
    n_slow_updates: int

    def delta_hr(self, hr: float) -> float | None:
        ref = self.hr_slow if self.hr_slow is not None else self.hr_fast
        if ref is None or hr <= 0:
            return None
        return hr - ref

    def expected_hr_band(self) -> tuple[float, float] | None:
        if self.hr_slow is None:
            return None
        margin = max(8.0, self.hr_slow * 0.12)
        return self.hr_slow - margin, self.hr_slow + margin


class PersonalBaseline:
    def __init__(self, path: Path | str = "user_baseline.json") -> None:
        self.path = Path(path)
        self._hr_fast: float | None = None
        self._hr_slow: float | None = None
        self._rmssd_fast: float | None = None
        self._rmssd_slow: float | None = None
        self._last_ts = 0.0
        self._n_slow = 0
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            with open(self.path) as f:
                d = json.load(f)
            self._hr_slow = d.get("hr_slow")
            self._rmssd_slow = d.get("rmssd_slow")
            self._n_slow = int(d.get("n_slow_updates", 0))
        except Exception:
            pass

    def _save(self) -> None:
        if self._hr_slow is None:
            return
        payload = {
            "hr_slow": self._hr_slow,
            "rmssd_slow": self._rmssd_slow,
            "n_slow_updates": self._n_slow,
            "updated_at": int(time.time()),
        }
        with open(self.path, "w") as f:
            json.dump(payload, f, indent=2)

    def observe(
        self,
        hr: float,
        rmssd: float,
        *,
        rest_like: bool,
        now: float | None = None,
    ) -> None:
        if not rest_like or hr <= 0:
            return
        now = now or time.time()
        dt = now - self._last_ts if self._last_ts > 0 else 30.0
        self._last_ts = now

        self._hr_fast = _ema_step(self._hr_fast, hr, dt, TAU_FAST_SEC)
        if rmssd > 0:
            self._rmssd_fast = _ema_step(self._rmssd_fast, rmssd, dt, TAU_FAST_SEC)

        self._hr_slow = _ema_step(self._hr_slow, hr, dt, TAU_SLOW_SEC)
        if rmssd > 0:
            self._rmssd_slow = _ema_step(self._rmssd_slow, rmssd, dt, TAU_SLOW_SEC)
        self._n_slow += 1
        if self._n_slow % 10 == 0:
            self._save()

    @property
    def slow_ready(self) -> bool:
        return self._n_slow >= MIN_SLOW_SAMPLES and self._hr_slow is not None

    def snapshot(self) -> BaselineSnapshot:
        return BaselineSnapshot(
            hr_fast=self._hr_fast,
            hr_slow=self._hr_slow,
            rmssd_fast=self._rmssd_fast,
            rmssd_slow=self._rmssd_slow,
            slow_ready=self.slow_ready,
            n_slow_updates=self._n_slow,
        )

    def to_mqtt(self, hr_now: float = 0.0) -> dict:
        snap = self.snapshot()
        band = snap.expected_hr_band()
        delta = snap.delta_hr(hr_now) if hr_now > 0 else None
        out = {
            "baseline_slow_ready": snap.slow_ready,
            "hr_baseline_fast": round(snap.hr_fast, 1) if snap.hr_fast else None,
            "hr_baseline_slow": round(snap.hr_slow, 1) if snap.hr_slow else None,
            "rmssd_baseline_fast": round(snap.rmssd_fast, 1) if snap.rmssd_fast else None,
            "rmssd_baseline_slow": round(snap.rmssd_slow, 1) if snap.rmssd_slow else None,
            "hr_delta_slow": round(delta, 1) if delta is not None else None,
        }
        if band:
            out["hr_band_lo"] = round(band[0], 1)
            out["hr_band_hi"] = round(band[1], 1)
        return out
