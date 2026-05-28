"""
Build-time toggles for the parallel research / legacy engines.

These engines run ALONGSIDE production for the thesis demonstrations
(negative-results / comparisons). They are published only on `biofizic/legacy/*`
and NEVER feed `biofizic/state` (the VR decision). Default OFF keeps production
light and free of the research-only dependencies (scipy / scikit-learn). Flip
a flag here and rebuild to enable for a specific research session.

Why default OFF:
  WESAD is a chest-strap-trained RF (RespiBAN ECG + EDA + EMG + RESP);
  applying it to wrist-only PPG without EDA is biased by design and produces
  fake-confident p_stress numbers. Documented as a negative result.
  VALENCE is an ad-hoc heuristic on PPA-z; also a documented negative result.
  PPG_PEAKS and RAW_PPG are required infrastructure for the above; they have
  no consumer once WESAD + VALENCE are off, so they're disabled too.
"""

from __future__ import annotations

# Parse raw PPG from the watch payload (requires PUBLISH_RAW_PPG on the watch).
ENABLE_RAW_PPG = False
# Server-side PPG peak detection over the raw PPG window (requires scipy).
ENABLE_PPG_PEAKS = False
# Parallel WESAD RandomForest stress probability (requires scikit-learn + model).
ENABLE_WESAD = False
# Ad-hoc valence heuristic (documented negative result; needs raw-PPG PPA).
ENABLE_VALENCE = False


def any_enabled() -> bool:
    return ENABLE_RAW_PPG or ENABLE_PPG_PEAKS or ENABLE_WESAD or ENABLE_VALENCE
