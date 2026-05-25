"""Per-user STILL baseline for motion feature normalization (no reset on motion)."""

from __future__ import annotations

from collections import deque

import numpy as np

from biofizic.config import PASSIVE_EMA_ALPHA, PASSIVE_REST_EPOCHS_MIN

STILL_CLASS = "STILL"


class MotionCalibrator:
    """
    Lock-in median/IQR on first STILL epochs, then slow EMA updates on STILL only.
    Never resets when the user scrolls or walks.
    """

    def __init__(self) -> None:
        self._warmup: deque[np.ndarray] = deque(maxlen=PASSIVE_REST_EPOCHS_MIN)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None
        self.ready = False
        self.n_still = 0

    def observe(self, features: np.ndarray, predicted_class: str) -> None:
        if predicted_class != STILL_CLASS:
            return
        x = features.flatten().copy()
        self.n_still += 1

        if not self.ready:
            self._warmup.append(x)
            if len(self._warmup) < PASSIVE_REST_EPOCHS_MIN:
                return
            mat = np.stack(list(self._warmup))
            self._mean = np.median(mat, axis=0)
            q75 = np.percentile(mat, 75, axis=0)
            q25 = np.percentile(mat, 25, axis=0)
            self._std = (q75 - q25) * 0.7413 + 1e-6
            self.ready = True
            return

        alpha = PASSIVE_EMA_ALPHA
        assert self._mean is not None and self._std is not None
        self._mean = (1.0 - alpha) * self._mean + alpha * x
        dev = np.abs(x - self._mean)
        self._std = (1.0 - alpha) * self._std + alpha * np.maximum(dev, 1e-6)

    def normalize(self, features: np.ndarray) -> np.ndarray:
        x = features.flatten()
        if self.ready and self._mean is not None and self._std is not None:
            return (x - self._mean) / self._std
        return x

    @property
    def n_samples(self) -> int:
        return self.n_still
