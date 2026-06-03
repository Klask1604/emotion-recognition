"""
Valence research tracks — one self-contained bundle per valence model.

The three valence models (WESAD / EEVR / CASE) are the SAME engine class with a
different `.joblib` bundle; each needs its own per-subject calibration so the
dashboard shows subject-relative emotion (neutral measured, not assumed 0). That
calibration is three small stateful objects that used to be spelled out by name in
the pipeline (`valence_baseline_eevr`, `valence_smoother_eevr`, ...) and reset one
by one in `reset_baseline` — adding a model meant editing the production engine in
two places (OCP violation + shotgun surgery).

A `ValenceTrack` packs those three objects (personal baseline + 60 s smoother +
verdict stabiliser) behind one key. The pipeline holds a dict
`{key: ValenceTrack(key)}` built from `ENABLED_VALENCE_MODELS`, so adding DEAP is a
single list entry — the engine never learns the model names. Lives under research/
because none of this feeds the production arousal verdict; it only enriches the
`biofizic/legacy/valence_*` dashboards (WESAD additionally caches for the watch's
2D verdict, which is research-on-trial by design).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from affectus.config import data_dir, valence_baseline_path
from affectus.dsp.valence_baseline import ValenceBaselineStore, ValenceSmoother
from affectus.dsp.emotion import ValenceVerdictStabilizer

# The valence models that run observed-only alongside production. WESAD is the
# one whose calibrated valence the watch's 2D verdict reads (still on trial); EEVR
# and CASE are comparison-only. Add a model here (after its .joblib exists) and the
# pipeline picks it up with zero engine edits.
ENABLED_VALENCE_MODELS = ["wesad", "eevr", "case"]

# Rolling-mean window for the published personal valence (matches the arousal w60
# decision window so the circumplex has uniform inertia on both axes).
VALENCE_SMOOTHER_WINDOW_S = 60.0


def _baseline_for(key: str) -> ValenceBaselineStore:
    """WESAD uses the canonical valence_baseline.json (it is the primary model);
    every other model gets its own file so each is re-centred independently."""
    if key == "wesad":
        return ValenceBaselineStore(persist=True)
    return ValenceBaselineStore(path=data_dir() / f"valence_baseline_{key}.json")


@dataclass
class ValenceTrack:
    """Per-subject calibration state for one valence model."""

    key: str
    baseline: ValenceBaselineStore = field(init=False)
    smoother: ValenceSmoother = field(init=False)
    stabilizer: ValenceVerdictStabilizer = field(init=False)

    def __post_init__(self) -> None:
        self.baseline = _baseline_for(self.key)
        self.smoother = ValenceSmoother(window_s=VALENCE_SMOOTHER_WINDOW_S)
        self.stabilizer = ValenceVerdictStabilizer()

    def reset(self, reported_valence: float | None = None) -> None:
        """Re-anchor on a recalibration: clear the resting samples + smoother. The
        neutral is re-measured from the new resting epochs; reported_valence only
        feeds the polarity sign-guard."""
        self.baseline.reset_for_recalibration(reported_valence)
        self.smoother.reset()


def build_valence_tracks() -> dict[str, ValenceTrack]:
    """The pipeline's single source of valence calibration state."""
    return {key: ValenceTrack(key) for key in ENABLED_VALENCE_MODELS}
