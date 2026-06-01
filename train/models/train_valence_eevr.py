#!/usr/bin/env python3
"""Train the EEVR valence model for the live comparison channel.

EEVR (VR-induced, 37 subjects, Empatica E4 BVP) valence axis = HVHA vs LVHA, i.e.
both HIGH-arousal so the contrast isolates valence (the WESAD trick). On the
research test this is ~56% LOSO (≈ chance) — VR design-truth labels do not carry
cross-subject valence in PPG. We still ship a bundle so it runs LIVE next to WESAD
and CASE on the watch, for the comparison dashboard.

Uses the SAME 33-feature shared extractor as WESAD/CASE (Xb in the npz cache), so
the bundle is drop-in compatible with affectus/legacy/valence_wesad_engine.py
(load it with model_path=models/valence_eevr.joblib).

Reads the cached features data/eevr_stratified.npz (produced by
train/eevr_valence_stratified.py) — no re-extraction.

Usage: ./venv/Scripts/python.exe train/train_valence_eevr.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from affectus.legacy.valence_features import VALENCE_FEATURE_NAMES  # noqa: E402

CACHE = ROOT / "data" / "eevr_stratified.npz"
OUT = ROOT / "models" / "valence_eevr.joblib"


def main() -> None:
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score
    import joblib

    d = np.load(CACHE, allow_pickle=True)
    quad, subj = d["quad"], d["subjects"]
    Xb = d["Xb"]
    # HVHA vs LVHA: both high-arousal, isolate valence (1 = positive = HVHA).
    mask = np.isin(quad, ("HVHA", "LVHA"))
    rows = mask & np.all(np.isfinite(Xb), axis=1)
    X = Xb[rows]
    y = (quad[rows] == "HVHA").astype(int)
    subj = subj[rows].astype(int)
    print(f"n={len(y)}, subjects={len(np.unique(subj))}, positive(HVHA)={y.mean():.1%}")

    # Honest LOSO score (report only).
    logo = LeaveOneGroupOut()
    baccs = []
    for tr, te in logo.split(X, y, groups=subj):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        baccs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    print(f"LOSO balanced accuracy: {np.mean(baccs):.1%} (± {np.std(baccs):.1%})")

    # Final model on all subjects, with probability for the live channel.
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
        "trained_on": "EEVR HVHA-vs-LVHA (VR valence axis, arousal-matched)",
    }, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
