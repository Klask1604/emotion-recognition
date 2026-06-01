#!/usr/bin/env python3
"""Train the CASE valence model for the live comparison channel.

CASE (continuous valence+arousal joystick, 30 subjects, BVP) is the ONLY dataset
where valence survives arousal-matching: HV(val>=7) vs LV(val<=3) inside the
HIGH-arousal band (arousal>=6), arousal equalized by bin-subsampling, reaches
~58% LOSO (p=0.014, real signal — driven by hrv_lf_hf, not arousal). This is the
strongest of the three valence models, so it runs LIVE next to WESAD and EEVR.

Uses the SAME 33-feature shared extractor (Xb in the npz cache) -> bundle is
drop-in compatible with affectus/legacy/valence_wesad_engine.py
(model_path=models/valence_case.joblib).

Reads cached features data/case_stratified.npz (from train/case_valence_stratified.py).

Usage: ./venv/Scripts/python.exe train/train_valence_case.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from affectus.legacy.valence_features import VALENCE_FEATURE_NAMES  # noqa: E402
from train.case_valence_stratified import HV_THR, LV_THR, HI_AROUSAL, _match_arousal  # noqa: E402

CACHE = ROOT / "data" / "case_stratified.npz"
OUT = ROOT / "models" / "valence_case.joblib"


def main() -> None:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score
    import joblib

    d = np.load(CACHE, allow_pickle=True)
    val, aro, subj = d["valence"], d["arousal"], d["subjects"]
    Xb = d["Xb"]
    # HV vs LV within the high-arousal band; then equalize arousal across classes
    # so the model learns valence, not the residual arousal difference.
    band = aro >= HI_AROUSAL
    hv, lv = val >= HV_THR, val <= LV_THR
    sel = band & (hv | lv) & np.all(np.isfinite(Xb), axis=1)
    X = Xb[sel]
    y = hv[sel].astype(int)
    a = aro[sel]
    g = subj[sel].astype(int)
    keep = _match_arousal(a, y, np.random.default_rng(0))
    X, y, g = X[keep], y[keep], g[keep]
    print(f"n={len(y)}, subjects={len(np.unique(g))}, positive(HV)={y.mean():.1%}")

    # Honest LOSO score (report only).
    logo = LeaveOneGroupOut()
    baccs = []
    for tr, te in logo.split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        baccs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    print(f"LOSO balanced accuracy: {np.mean(baccs):.1%} (± {np.std(baccs):.1%})")

    # Final model on all matched windows, with probability for the live channel.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xn = (X - mean) / std
    final = SVC(kernel="rbf", C=1.0, gamma="scale",
                class_weight="balanced", probability=True)
    final.fit(Xn, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_names": list(VALENCE_FEATURE_NAMES),
        "feature_mean": mean,
        "feature_std": std,
        "classes": ["negative", "positive"],
        "trained_on": "CASE HV-vs-LV high-arousal matched (continuous joystick valence)",
    }, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
