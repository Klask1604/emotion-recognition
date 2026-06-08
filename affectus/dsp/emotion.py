"""
Emotion verdict from the Russell circumplex (1980): a quadrant of AROUSAL × VALENCE.

Both inputs are PERSONAL z-scores centred on the subject's own resting baseline
(0 = neutral), so the verdict is subject-relative. The neutral dead-band around 0
on the valence axis keeps a near-zero (or pre-calibration) valence from being
forced into a quadrant, honest about the weak valence signal at low affect
intensity.

This is the single source of truth for the verdict: the compute service publishes
it on biofizic/state(/live) for the watch, and it mirrors the CASE logic in the
Grafana circumplex dashboard so the two never disagree. Pure function, no I/O.

Codes match the dashboard's value-mappings:
    0 Neutru · 1 Calm · 2 Trist · 3 Bucuros · 4 Stresat
"""

from __future__ import annotations

from collections import deque

# (label, code), Romanian single words, consistent with arousal_scale_10_to_label.
NEUTRAL = ("Neutru", 0)
CALM = ("Calm", 1)        # low arousal, positive valence (calm/relaxed)
SAD = ("Trist", 2)        # low arousal, negative valence (sad/bored)
HAPPY = ("Bucuros", 3)    # high arousal, positive valence (happy/excited)
STRESSED = ("Stresat", 4)  # high arousal, negative valence (stress/anger)


def emotion_verdict(
    arousal_z: float,
    valence_personal: float,
    deadband: float,
    valence_ready: bool,
) -> tuple[str, int]:
    """Map (arousal z, personal valence, dead-band) to an emotion (label, code).

    arousal_z:        personal arousal z-score (>=0 = at/above resting baseline).
    valence_personal: personal valence (>0 = more positive than resting neutral).
    deadband:         neutral-zone half-width on the valence axis (subject-derived).
    valence_ready:    whether the valence baseline has locked; if not, valence is
                      not trustworthy and the verdict falls back to Neutral.
    """
    if not valence_ready or abs(valence_personal) < deadband:
        return NEUTRAL
    if arousal_z >= 0.0:
        return HAPPY if valence_personal >= 0.0 else STRESSED
    return CALM if valence_personal >= 0.0 else SAD


class ValenceVerdictStabilizer:
    """Brings the valence verdict to processing parity with arousal.

    The raw 60 s-smoothed valence still flips the emotion quadrant epoch-to-epoch
    on a low-SNR subject (observed: 40% transitions while sitting still). Arousal
    avoids this with Kalman smoothing + a quality gate + display hysteresis; this
    applies the same three guards to valence:

      1. CONFIDENCE GATE, the WESAD model's margin |2p-1| (~0.5 = chance). When
         it is below CONF_MIN the epoch is too uncertain to assert a sign, so the
         valence used for the verdict is pulled toward 0 (neutral), exactly like a
         low-quality arousal epoch barely moving the Kalman estimate.
      2. MEDIAN SMOOTHING, a rolling median over the last N epochs resists the
         occasional spike better than a mean (one outlier can't drag it).
      3. VERDICT HYSTERESIS, the displayed quadrant only changes after a
         different one persists for HYSTERESIS_TICKS consecutive epochs, mirroring
         _update_live_arousal_hysteresis so a single noisy epoch never flips it.
    """

    CONF_MIN = 0.7          # below this margin the epoch is treated as neutral
                            # (raised from 0.6: at ~0.5 the WESAD model is at
                            # chance on this subject, so only clearly-confident
                            # epochs assert a quadrant, honest "don't know" else)
    MEDIAN_WINDOW = 5       # epochs in the rolling median
    HYSTERESIS_TICKS = 3    # consecutive epochs a new quadrant must persist

    def __init__(self) -> None:
        self._vals: deque[float] = deque(maxlen=self.MEDIAN_WINDOW)
        self._displayed: int | None = None
        self._candidate: int | None = None
        self._streak: int = 0

    def reset(self) -> None:
        self._vals.clear()
        self._displayed = None
        self._candidate = None
        self._streak = 0

    @staticmethod
    def _median(values: list[float]) -> float:
        s = sorted(values)
        n = len(s)
        mid = n // 2
        return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])

    def update(
        self,
        arousal_z: float,
        valence_personal: float,
        confidence: float,
        deadband: float,
        valence_ready: bool,
    ) -> tuple[str, int]:
        """Fold one epoch's valence into the stabilised verdict and return it."""
        # 1) Confidence gate: an uncertain epoch contributes ~0 (neutral), so it
        # neither pushes a quadrant nor pollutes the median.
        gated = valence_personal if confidence >= self.CONF_MIN else 0.0
        # 2) Median smoothing over the recent gated values.
        self._vals.append(gated)
        v_med = self._median(list(self._vals))
        # Raw quadrant on the smoothed value.
        _, raw_code = emotion_verdict(arousal_z, v_med, deadband, valence_ready)
        # 3) Hysteresis on the quadrant code (same logic as live arousal).
        if self._displayed is None:
            self._displayed = raw_code
        elif raw_code == self._displayed:
            self._candidate = None
            self._streak = 0
        elif raw_code == self._candidate:
            self._streak += 1
            if self._streak >= self.HYSTERESIS_TICKS:
                self._displayed = raw_code
                self._candidate = None
                self._streak = 0
        else:
            self._candidate = raw_code
            self._streak = 1
        return _label_for_code(self._displayed)


_CODE_LABEL = {0: "Neutru", 1: "Calm", 2: "Trist", 3: "Bucuros", 4: "Stresat"}


def _label_for_code(code: int) -> tuple[str, int]:
    return _CODE_LABEL.get(code, "Neutru"), code
