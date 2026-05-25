"""
Unified resting baseline store.

Rules:
  1. Lock-in after STILL_EPOCHS_BEFORE_BASELINE_LOCK STILL observations (median).
  2. Update with slow EMA only during STILL (never on HAND/WALK/SCROLL).
  3. Never reset on motion change. Reset only via explicit recalibrate command.
  4. Persist to data/rest_baseline.json
"""

from __future__ import annotations

import json
import logging
from collections import deque
from pathlib import Path

import numpy as np

from biofizic.config import (
    BASELINE_EMA_ALPHA,
    STILL_EPOCHS_BEFORE_BASELINE_LOCK,
    rest_baseline_path,
)

log = logging.getLogger("rest_baseline")


class RestBaselineStore:
    def __init__(self, path: Path | None = None) -> None:
        self._path = path or rest_baseline_path()
        self._warmup_rmssd: deque[float] = deque(maxlen=STILL_EPOCHS_BEFORE_BASELINE_LOCK)
        self._warmup_stress_index: deque[float] = deque(
            maxlen=STILL_EPOCHS_BEFORE_BASELINE_LOCK
        )
        self._baseline_rmssd_ms: float | None = None
        self._baseline_stress_index: float | None = None
        self._rmssd_scale: float = 15.0
        self.is_ready = False
        self.still_observation_count = 0
        self._load()

    def observe_still(
        self,
        rmssd_ms: float,
        kubios_stress_index: float,
    ) -> None:
        """Update baseline only when user is STILL."""
        if rmssd_ms <= 0 or kubios_stress_index <= 0:
            return
        self.still_observation_count += 1

        if not self.is_ready:
            self._warmup_rmssd.append(rmssd_ms)
            self._warmup_stress_index.append(kubios_stress_index)
            if len(self._warmup_rmssd) < STILL_EPOCHS_BEFORE_BASELINE_LOCK:
                return
            self._baseline_rmssd_ms = float(np.median(self._warmup_rmssd))
            self._baseline_stress_index = float(np.median(self._warmup_stress_index))
            self._rmssd_scale = max(
                8.0,
                float(np.std(self._warmup_rmssd))
                if len(self._warmup_rmssd) > 2
                else 15.0,
            )
            self.is_ready = True
            self._save()
            log.info(
                "Baseline locked: stress_index=%.2f rmssd=%.1f",
                self._baseline_stress_index,
                self._baseline_rmssd_ms,
            )
            return

        assert self._baseline_rmssd_ms is not None
        assert self._baseline_stress_index is not None
        alpha = BASELINE_EMA_ALPHA
        self._baseline_rmssd_ms = (1 - alpha) * self._baseline_rmssd_ms + alpha * rmssd_ms
        self._baseline_stress_index = (
            (1 - alpha) * self._baseline_stress_index + alpha * kubios_stress_index
        )
        self._rmssd_scale = (1 - alpha) * self._rmssd_scale + alpha * max(
            8.0, abs(self._baseline_rmssd_ms - rmssd_ms)
        )
        self._save()

    def reset_for_recalibration(self) -> None:
        """Only call from biofizic/cmd/calibrate."""
        self._warmup_rmssd.clear()
        self._warmup_stress_index.clear()
        self._baseline_rmssd_ms = None
        self._baseline_stress_index = None
        self._rmssd_scale = 15.0
        self.is_ready = False
        self.still_observation_count = 0
        self._save()
        log.info("Baseline reset for recalibration")

    def stress_index_z_score(self, kubios_stress_index: float) -> float:
        base = self._baseline_stress_index if self._baseline_stress_index else 10.0
        scale = max(0.5, base * 0.15)
        return float(np.clip((kubios_stress_index - base) / scale, -3.0, 3.0))

    def rmssd_z_score(self, rmssd_ms: float) -> float:
        base = self._baseline_rmssd_ms if self._baseline_rmssd_ms else 45.0
        scale = max(8.0, self._rmssd_scale)
        return float(np.clip((base - rmssd_ms) / scale, -3.0, 3.0))

    @property
    def baseline_stress_index(self) -> float | None:
        return self._baseline_stress_index

    @property
    def baseline_rmssd_ms(self) -> float | None:
        return self._baseline_rmssd_ms

    def _save(self) -> None:
        payload = {
            "is_ready": self.is_ready,
            "still_observation_count": self.still_observation_count,
            "baseline_stress_index": self._baseline_stress_index,
            "baseline_rmssd_ms": self._baseline_rmssd_ms,
            "rmssd_scale": self._rmssd_scale,
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
            self.still_observation_count = int(data.get("still_observation_count", 0))
            self._baseline_stress_index = data.get("baseline_stress_index")
            self._baseline_rmssd_ms = data.get("baseline_rmssd_ms")
            self._rmssd_scale = float(data.get("rmssd_scale", 15.0))
        except Exception as exc:
            log.warning("Could not load baseline file: %s", exc)
