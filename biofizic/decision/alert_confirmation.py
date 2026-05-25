"""Alert confirmation and RMSSD spike filter."""

from __future__ import annotations

from collections import deque

import numpy as np

from biofizic.config import (
    ALERT_CONFIRMATION_EPOCH_COUNT,
    REST_ACCELERATION_P90_MAX,
    RMSSD_SPIKE_RATIO_THRESHOLD,
    RMSSD_Z_SUPPRESS_ALERT,
    STRESS_INDEX_Z_ALERT,
    STRESS_INDEX_Z_ALERT_STRONG,
)
from biofizic.context_engine import ActivityContext
from biofizic.decision.arousal_mapper import (
    kubios_zone_for_stress_index,
    zone_is_alert_or_higher,
    zone_is_elevated_or_higher,
)


class AlertConfirmationGate:
    """
    Require 2 consecutive elevated epochs for full Alert.
    At rest, require dual agreement of stress_index z-score and RMSSD z-score.
    """

    def __init__(self) -> None:
        self._elevated_streak = 0
        self._recent_rmssd: deque[float] = deque(maxlen=4)

    def apply(
        self,
        *,
        kubios_stress_index: float,
        rmssd_ms: float,
        stress_index_z_score: float,
        rmssd_z_score: float,
        activity: ActivityContext,
        arousal: float,
        arousal_scale_10: int,
    ) -> tuple[float, int, str]:
        self._recent_rmssd.append(rmssd_ms)
        zone = kubios_zone_for_stress_index(kubios_stress_index)
        mode = "kubios_zone"

        if (
            len(self._recent_rmssd) >= 2
            and activity.is_rest_like
            and activity.acc_p90 <= REST_ACCELERATION_P90_MAX
            and rmssd_ms > 0
        ):
            previous = list(self._recent_rmssd)[:-1]
            median = float(np.median(previous)) if previous else rmssd_ms
            if median > 0 and abs(rmssd_ms - median) / median > RMSSD_SPIKE_RATIO_THRESHOLD:
                cap = 0.62
                return min(arousal, cap), min(arousal_scale_10, 6), "rmssd_spike_cap"

        if not zone_is_elevated_or_higher(zone):
            self._elevated_streak = 0
            return arousal, arousal_scale_10, mode

        if activity.is_rest_like:
            dual_ok = (
                stress_index_z_score >= STRESS_INDEX_Z_ALERT
                and rmssd_z_score >= RMSSD_Z_SUPPRESS_ALERT
            ) or stress_index_z_score >= STRESS_INDEX_Z_ALERT_STRONG
            if not dual_ok:
                self._elevated_streak = 0
                return min(arousal, 0.62), min(arousal_scale_10, 6), "rest_dual_veto"

        self._elevated_streak += 1
        if zone_is_alert_or_higher(zone):
            if self._elevated_streak < ALERT_CONFIRMATION_EPOCH_COUNT:
                return min(arousal, 0.62), min(arousal_scale_10, 6), "alert_pending"
            mode = "alert_confirmed"
        return arousal, arousal_scale_10, mode
