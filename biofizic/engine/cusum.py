"""One-sided CUSUM change detector for sustained stress elevation.

Replaces the "N consecutive elevated epochs" counter with a sequential
change-point test (Page 1954; Montgomery, Statistical Quality Control). Run on
the personal stress-index z-score stream:

    S_t = max(0, S_{t-1} + (z_t - k))     alert when S_t > h

k is the reference value (slack) that the signal must persistently exceed
before evidence accumulates; h is the decision interval. Standard SPC defaults
k ~= 0.5 and h ~= 4-5 (in sigma units, which is exactly what z is). The alert
latches while S_t stays above the threshold and clears once the accumulator
returns to zero, giving change detection with built-in hysteresis.
"""

from __future__ import annotations

from dataclasses import dataclass

from biofizic.config import CUSUM_SLACK_K, CUSUM_THRESHOLD_H


@dataclass
class CusumDetector:
    k: float = CUSUM_SLACK_K
    h: float = CUSUM_THRESHOLD_H
    s_hi: float = 0.0
    alert: bool = False

    def update(self, z: float) -> bool:
        self.s_hi = max(0.0, self.s_hi + (float(z) - self.k))
        if self.s_hi > self.h:
            self.alert = True
        elif self.s_hi == 0.0:
            self.alert = False
        return self.alert

    def reset(self) -> None:
        self.s_hi = 0.0
        self.alert = False
