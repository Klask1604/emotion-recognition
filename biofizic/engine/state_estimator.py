"""
Scalar Kalman smoother for the personal stress z-score.

The true autonomic state is modelled as a slowly drifting latent variable; each
30 s epoch yields a noisy measurement of the personal stress z. The measurement
variance scales inversely with the signal quality Q, so a low-quality epoch
(many IBI artifacts / motion) has a huge measurement variance, a tiny Kalman
gain, and therefore barely moves the estimate. This unifies the signal-quality
gate and the earlier "hold last value" patch into one principled filter
(Kalman 1960; state-space HRV).

    predict:  P <- P + q
    update:   K = P / (P + r),   x <- x + K (z - x),   P <- (1 - K) P
    with r = MEAS_VAR_BASE / max(Q, QUALITY_FLOOR)
"""

from __future__ import annotations

from dataclasses import dataclass

from biofizic.config import (
    KALMAN_MEAS_VAR_BASE,
    KALMAN_PROCESS_VAR,
    KALMAN_QUALITY_FLOOR,
)


@dataclass
class StressStateEstimator:
    """Per-session Kalman state for the personal stress z."""

    x: float = 0.0   # estimated latent z (0 = at personal baseline)
    P: float = 1.0   # estimate variance
    process_var: float = KALMAN_PROCESS_VAR

    def measurement_variance(self, quality: float) -> float:
        q = max(float(quality), KALMAN_QUALITY_FLOOR)
        return KALMAN_MEAS_VAR_BASE / q

    def update(self, z_measured: float, quality: float) -> tuple[float, float]:
        """Fold one epoch's z into the estimate. Returns (x_filtered, gain)."""
        r = self.measurement_variance(quality)
        self.P += self.process_var
        gain = self.P / (self.P + r)
        self.x += gain * (float(z_measured) - self.x)
        self.P *= 1.0 - gain
        return self.x, gain

    def value(self) -> float:
        return self.x

    def reset(self) -> None:
        self.x = 0.0
        self.P = 1.0
