"""
Valence heuristic — DOCUMENTED NEGATIVE RESULT (legacy / research only).

    valence = clip(0.55 * tanh(RMSSD_z / 2) - 0.35 * tanh(PPA_z / 2), -1, 1)

The coefficients 0.55 / 0.35 are ad-hoc — tuned on a handful of recordings, with
NO support in the HRV literature. This engine exists ONLY to show, with data
(see the Grafana "Valence demo" dashboard), that wrist PPG amplitude does not
separate positive from negative affect and that the axis is too noisy/collapses
under motion. It is published on biofizic/legacy/valence and NEVER feeds the
production decision. Do not present these coefficients as validated.
"""

from __future__ import annotations

import math

# Ad-hoc, unvalidated weights — kept here (not in config) precisely because they
# are not a defensible parameter of the production system.
_RMSSD_COEF = 0.55
_PPA_COEF = 0.35


class ValenceEngine:
    def compute(self, *, rmssd_z: float, ppa_z: float) -> dict:
        v = _RMSSD_COEF * math.tanh(rmssd_z / 2.0) - _PPA_COEF * math.tanh(ppa_z / 2.0)
        v = max(-1.0, min(1.0, v))
        return {
            "valence": round(v, 3),
            "rmssd_z": round(rmssd_z, 2),
            "ppa_z": round(ppa_z, 2),
        }
