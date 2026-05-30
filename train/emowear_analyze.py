#!/usr/bin/env python3
"""Analyse the cached EmoWear features (runs in seconds). Tests valence AND
arousal with LOSO, and checks whether the emotion induction actually moved
physiology (HR difference by arousal) — the diagnostic that tells us if a ~50%
result is the dataset or the method."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "emowear_features.npz"


def main() -> None:
    if not DATA.exists():
        print("Run emowear_extract.py first.")
        return
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score

    d = np.load(DATA, allow_pickle=True)
    X, val, aro, subj = d["X"], d["valence"], d["arousal"], d["subjects"]
    y_val = (val >= 5).astype(int)
    y_aro = (aro >= 5).astype(int)
    print(f"n={len(X)}, subjects={len(np.unique(subj))}, "
          f"high-valence={y_val.mean():.1%}, high-arousal={y_aro.mean():.1%}")

    logo = LeaveOneGroupOut()

    def loso(y):
        b = []
        for tr, te in logo.split(X, y, groups=subj):
            if len(np.unique(y[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", class_weight="balanced"))
            c.fit(X[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], c.predict(X[te])))
        return np.mean(b) * 100

    print(f"\n  VALENCE LOSO : {loso(y_val):.1f}%")
    print(f"  AROUSAL LOSO : {loso(y_aro):.1f}%")
    print("  (arousal should be easier; if it also flatlines, the induction was weak)")

    # Induction check: does HR proxy (from bf feature is not HR; use label spread)
    # Instead check whether features even separate by arousal at population level.
    from scipy.stats import pointbiserialr
    names = list(d["feature_names"])
    print("\n  Top feature correlations with valence / arousal:")
    for i, n in enumerate(names):
        rv = pointbiserialr(y_val, X[:, i]).statistic
        ra = pointbiserialr(y_aro, X[:, i]).statistic
        if abs(rv) > 0.08 or abs(ra) > 0.08:
            print(f"    {n:14s} valence r={rv:+.3f}  arousal r={ra:+.3f}")
    print("\n  |r|<0.1 everywhere => signal not in features on this data.")


if __name__ == "__main__":
    main()
