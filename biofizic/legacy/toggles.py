"""
Build-time toggles for the parallel research / legacy engines.

These engines run ALONGSIDE production for the thesis demonstrations
(negative-results / comparisons). They are published only on `biofizic/legacy/*`
and NEVER feed `biofizic/state` (the VR decision). Default off keeps production
light and free of the research-only dependencies (scipy / scikit-learn).

Flip a flag here and rebuild to enable (a UI toggle could come later).
"""

from __future__ import annotations

# Parse raw PPG from the watch payload (requires PUBLISH_RAW_PPG on the watch).
ENABLE_RAW_PPG = True
# Server-side PPG peak detection over the raw PPG window (requires scipy).
ENABLE_PPG_PEAKS = True
# Parallel WESAD RandomForest stress probability (requires scikit-learn + model).
ENABLE_WESAD = True
# Ad-hoc valence heuristic (documented negative result; needs raw-PPG PPA).
ENABLE_VALENCE = True


def any_enabled() -> bool:
    return ENABLE_RAW_PPG or ENABLE_PPG_PEAKS or ENABLE_WESAD or ENABLE_VALENCE
