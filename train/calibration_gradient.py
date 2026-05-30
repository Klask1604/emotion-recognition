#!/usr/bin/env python3
"""Calibration gradient: for each cached dataset, measure how accuracy changes
across three personalisation regimes, to decide what the product (live system)
should do.

  1. LOSO (zero calibration)      -> universal model, new user, nothing personal
  2. LOSO + few-shot calibration  -> universal model + a short PASSIVE baseline
                                     from the new user (subject-wise normalisation
                                     fitted on the first K windows of the test
                                     subject, applied to the rest). Mirrors the
                                     live RestBaselineStore.
  3. Within-subject               -> train+test on the SAME subject's own data
                                     (temporal split). This is what David's thesis
                                     and most "high accuracy" PPG papers do; it is
                                     the optimistic upper bound (the model has seen
                                     the subject).

Run: ./venv/Scripts/python.exe train/calibration_gradient.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]

from sklearn.svm import SVC  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402
from sklearn.metrics import balanced_accuracy_score  # noqa: E402

DATASETS = {
    "WESAD": None,   # built specially below (stress vs amusement)
    "CASE": ROOT / "data" / "case_features.npz",
    "CASE_v2": ROOT / "data" / "case_features_v2.npz",
    "EmoWear": ROOT / "data" / "emowear_features.npz",
}

CALIB_FRAC = 0.30   # first 30% of a test subject's windows used for calibration


def _clf():
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", class_weight="balanced"))


def loso_zero(X, y, subj):
    logo = LeaveOneGroupOut()
    b = []
    for tr, te in logo.split(X, y, groups=subj):
        if len(np.unique(y[tr])) < 2:
            continue
        c = _clf()
        c.fit(X[tr], y[tr])
        b.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return np.mean(b) * 100 if b else 0


def loso_fewshot(X, y, subj):
    """Train on N-1 subjects (each z-scored by its own stats). For the held-out
    subject, use the first CALIB_FRAC of its windows ONLY to estimate that
    subject's personal mean/std (passive baseline), z-score the rest with it, and
    test on the rest. The calibration windows are never used for testing, and the
    classifier never sees the test subject's labels — this is honest few-shot
    personalisation, the live-system scenario.

    Subjects are z-scored individually (train + test) so the classifier learns
    on a per-subject-normalised space, and the new user is mapped into the same
    space via its own short passive baseline."""
    logo = LeaveOneGroupOut()
    b = []
    for tr, te in logo.split(X, y, groups=subj):
        if len(np.unique(y[tr])) < 2:
            continue
        te = np.array(te)
        k = max(5, int(len(te) * CALIB_FRAC))
        calib, test = te[:k], te[k:]
        if len(test) < 5 or len(np.unique(y[tr])) < 2:
            continue

        # Train data: z-score each training subject by its own stats.
        Xtr = X[tr].copy()
        subj_tr = subj[tr]
        for s in np.unique(subj_tr):
            m = subj_tr == s
            Xtr[m] = (X[tr][m] - X[tr][m].mean(0)) / (X[tr][m].std(0) + 1e-9)

        # Test subject: personal stats from the calibration windows only.
        mu, sd = X[calib].mean(0), X[calib].std(0) + 1e-9
        Xte = (X[test] - mu) / sd

        c = SVC(kernel="rbf", class_weight="balanced").fit(Xtr, y[tr])
        b.append(balanced_accuracy_score(y[test], c.predict(Xte)))
    return np.mean(b) * 100 if b else 0


def within_subject(X, y, subj, frac=0.7):
    """Train on first frac of each subject, test on the rest of the SAME subject."""
    Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
    for s in np.unique(subj):
        m = np.where(subj == s)[0]
        k = int(len(m) * frac)
        Xtr_l.append(X[m[:k]]); ytr_l.append(y[m[:k]])
        Xte_l.append(X[m[k:]]); yte_l.append(y[m[k:]])
    Xtr, ytr = np.vstack(Xtr_l), np.concatenate(ytr_l)
    Xte, yte = np.vstack(Xte_l), np.concatenate(yte_l)
    if len(np.unique(ytr)) < 2:
        return 0
    c = _clf().fit(Xtr, ytr)
    return balanced_accuracy_score(yte, c.predict(Xte)) * 100


def run(name, X, val, aro, subj):
    for label_name, lab in [("valence", val), ("arousal", aro)]:
        y = (lab >= 5).astype(int)
        z = loso_zero(X, y, subj)
        f = loso_fewshot(X, y, subj)
        w = within_subject(X, y, subj)
        print(f"  {name:10s} {label_name:8s} | LOSO {z:5.1f}% | +few-shot {f:5.1f}% | within-subj {w:5.1f}%")


def main():
    print("Dataset    Label    | zero-calib | passive-calib | within-subject(David-style)")
    print("-" * 78)
    for name, path in DATASETS.items():
        if name == "WESAD":
            continue  # handled by wesad_healthcheck separately
        if path is None or not path.exists():
            continue
        d = np.load(path, allow_pickle=True)
        run(name, d["X"], d["valence"], d["arousal"], d["subjects"])


if __name__ == "__main__":
    main()
