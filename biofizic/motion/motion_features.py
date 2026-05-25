"""WISDM-aligned motion features from watch accelerometer and gyroscope stats."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

MOTION_FEATURE_NAMES = [
    "acc_rms",
    "acc_p90",
    "acc_std",
    "gyro_rms",
    "gyro_p90",
    "gyro_std",
    "gyro_acc_ratio",
    "acc_band_low",
    "acc_band_high",
]


@dataclass(frozen=True)
class MotionFeatureVector:
    """Fixed-length feature vector for HAR inference."""

    values: np.ndarray

    @classmethod
    def from_epoch_dict(cls, data: dict) -> MotionFeatureVector:
        acc_rms = float(data.get("acc_rms", 0))
        acc_p90 = float(data.get("acc_p90", acc_rms))
        acc_std = float(data.get("acc_std", 0))
        gyro_rms = float(data.get("gyro_rms", 0))
        gyro_p90 = float(data.get("gyro_p90", gyro_rms))
        gyro_std = float(data.get("gyro_std", 0))
        ratio = gyro_rms / max(acc_rms, 0.05)
        band_low = acc_rms * 0.6
        band_high = acc_p90 * 1.2
        return cls(
            values=np.array(
                [
                    acc_rms,
                    acc_p90,
                    acc_std,
                    gyro_rms,
                    gyro_p90,
                    gyro_std,
                    ratio,
                    band_low,
                    band_high,
                ],
                dtype=float,
            )
        )

    @classmethod
    def from_wisdm_window(
        cls,
        acc_mag: np.ndarray,
        gyro_mag: np.ndarray,
        *,
        sample_hz: float = 20.0,
    ) -> MotionFeatureVector | None:
        if len(acc_mag) < int(sample_hz * 5):
            return None
        acc_dyn = np.abs(acc_mag - np.median(acc_mag))
        gyro = gyro_mag if len(gyro_mag) == len(acc_mag) else gyro_mag[: len(acc_mag)]
        acc_rms = float(np.sqrt(np.mean(acc_dyn**2)))
        acc_p90 = float(np.percentile(acc_dyn, 90))
        acc_std = float(np.std(acc_dyn))
        gyro_rms = float(np.sqrt(np.mean(gyro**2))) if len(gyro) else 0.0
        gyro_p90 = float(np.percentile(gyro, 90)) if len(gyro) else 0.0
        gyro_std = float(np.std(gyro)) if len(gyro) else 0.0
        ratio = gyro_rms / max(acc_rms, 0.05)
        fft = np.abs(np.fft.rfft(acc_dyn))
        freqs = np.fft.rfftfreq(len(acc_dyn), d=1.0 / sample_hz)
        low = float(np.sum(fft[(freqs >= 0.5) & (freqs < 3.0)]))
        high = float(np.sum(fft[freqs >= 3.0]))
        return cls(
            values=np.array(
                [
                    acc_rms,
                    acc_p90,
                    acc_std,
                    gyro_rms,
                    gyro_p90,
                    gyro_std,
                    ratio,
                    low,
                    high,
                ],
                dtype=float,
            )
        )
