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
# ON: feeds the respiration comparator's PPG-amplitude arm (research data
# collection). Turn off for a pure-production build to drop PPG bandwidth/CPU.
ENABLE_RAW_PPG = True
# Server-side PPG peak detection over the raw PPG window (requires scipy).
# ON: PPG peaks + reconstructed-from-PPG IBI for the validation dashboard
# (compare Samsung IBI vs PPG-derived IBI). Research data collection.
ENABLE_PPG_PEAKS = True
# Parallel WESAD RandomForest stress probability (requires scikit-learn + model).
# OFF: avoids the model-file dependency; it is a documented negative result we
# can enable later if models/wesad_rf.joblib is present on the host.
ENABLE_WESAD = False
# Ad-hoc valence heuristic — SUPERSEDED by the PPG frequency-domain method
# (ENABLE_VALENCE_FD). Kept OFF as an archived negative result; the invented
# 0.55/0.35 coefficients have no literature basis. Re-enable only to reproduce
# the old comparison.
ENABLE_VALENCE = False
# Respiration comparator: RSA-from-IBI vs PPG-amplitude, side by side, on
# biofizic/legacy/resp. Research-only; decides which (if any) source is reliable
# enough to fuse later. Needs raw PPG on the watch for the PPG arm (RSA arm
# works from IBI alone).
# ON: collect RSA-vs-PPG respiration comparison data on biofizic/legacy/resp
# (validate which source is reliable on wrist for a slow breather). Research
# only; never feeds biofizic/state.
ENABLE_RESPIRATION_COMPARE = True
# PPG frequency-domain valence features (Frontiers 2025 replication) on
# biofizic/legacy/valence_fd. Features only, no verdict; for the thesis
# demonstration + future classifier training. Needs raw PPG. Never feeds state.
ENABLE_VALENCE_FD = True
# WESAD-trained valence model (stress vs amusement, ~85% LOSO) run as an
# OBSERVED-ONLY research engine on biofizic/legacy/valence_wesad. Same model and
# extractor as the production channel, but logged side-by-side with the verdict
# to judge cross-device transfer to the watch BEFORE it is trusted in the
# decision. Needs raw PPG + models/valence_wesad.joblib. Never feeds state.
ENABLE_VALENCE_WESAD = True
# EEVR- and CASE-trained valence models, run OBSERVED-ONLY alongside WESAD on
# biofizic/legacy/valence_eevr and biofizic/legacy/valence_case. Same 33-feature
# shared extractor, same engine class (just a different model bundle), so the
# three predictions can be compared live on the watch's own signal. Research /
# thesis comparison only — NEVER feeds biofizic/state. EEVR ≈ chance (56% LOSO),
# CASE ≈ 58% (the only real valence signal); the dashboard shows the divergence.
# Need raw PPG + models/valence_eevr.joblib / valence_case.joblib.
ENABLE_VALENCE_EEVR = True
ENABLE_VALENCE_CASE = True


def any_enabled() -> bool:
    return (
        ENABLE_RAW_PPG
        or ENABLE_PPG_PEAKS
        or ENABLE_WESAD
        or ENABLE_VALENCE
        or ENABLE_RESPIRATION_COMPARE
        or ENABLE_VALENCE_FD
        or ENABLE_VALENCE_WESAD
        or ENABLE_VALENCE_EEVR
        or ENABLE_VALENCE_CASE
    )
