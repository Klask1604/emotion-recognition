#!/usr/bin/env python3
"""Analyse cached CASE features (instant). Tests valence + arousal with LOSO,
plus the refinements (subject-wise normalisation, per-trial feature averaging,
clear-label filtering). CASE has continuous joystick annotation, so labels are
the cleanest of all the datasets — this is the decisive valence test."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "data" / "case_features.npz"


def main() -> None:
    if not DATA.exists():
        print("Run case_extract.py first.")
        return
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score

    d = np.load(DATA, allow_pickle=True)
    X, val, aro, subj, trials = d["X"], d["valence"], d["arousal"], d["subjects"], d["trials"]
    logo = LeaveOneGroupOut()

    def loso(Xm, y, g):
        b = []
        for tr, te in logo.split(Xm, y, groups=g):
            if len(np.unique(y[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", class_weight="balanced"))
            c.fit(Xm[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], c.predict(Xm[te])))
        return np.mean(b) * 100 if b else 0

    def subjnorm(Xm, g):
        out = Xm.copy()
        for s in np.unique(g):
            m = g == s
            out[m] = (Xm[m] - Xm[m].mean(0)) / (Xm[m].std(0) + 1e-9)
        return out

    yv = (val >= 5).astype(int)
    ya = (aro >= 5).astype(int)
    print(f"CASE: n={len(X)} windows, subjects={len(np.unique(subj))}, "
          f"high-valence={yv.mean():.1%}, high-arousal={ya.mean():.1%}\n")

    Xsw = subjnorm(X, subj)
    print("=== Per-window ===")
    print(f"  VALENCE (global norm)      : {loso(X, yv, subj):.1f}%")
    print(f"  VALENCE (subject-wise norm): {loso(Xsw, yv, subj):.1f}%")
    print(f"  AROUSAL (subject-wise norm): {loso(Xsw, ya, subj):.1f}%")

    # Clear labels only (drop neutral 4-6)
    mask = (val <= 3) | (val >= 7)
    print(f"  VALENCE (subj-norm, clear) : {loso(Xsw[mask], (val[mask] >= 5).astype(int), subj[mask]):.1f}%  (n={mask.sum()})")

    # Per-trial averaging
    tids = np.unique(trials)
    Xa = np.array([X[trials == t].mean(0) for t in tids])
    va = np.array([val[trials == t][0] for t in tids])
    sa = np.array([subj[trials == t][0] for t in tids])
    Xa_sw = subjnorm(Xa, sa)
    print("\n=== Per-trial averaging ===")
    print(f"  VALENCE (avg + subj-norm)  : {loso(Xa_sw, (va >= 5).astype(int), sa):.1f}%")


if __name__ == "__main__":
    main()
