#!/usr/bin/env python3
"""
Train a valence classifier on the DEAP PPG frequency-domain features with
HONEST leave-one-subject-out (LOSO) cross-validation — the validation the
literature warns is usually skipped (leave-one-point-out inflates accuracy by
letting the model memorise the subject).

We report BALANCED accuracy and F1, not raw accuracy, because the DEAP valence
split is imbalanced (~22% high valence): a model that always predicts "low"
would score ~78% raw accuracy while learning nothing.

Trains an SVM (RBF) like the reference paper, with per-fold standardisation.
Saves the final model (trained on all subjects) to models/deap_valence_fd.joblib
for the transfer test onto Galaxy-Watch features.

Usage:
    ./venv/Scripts/python.exe train/deap_train_valence.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "deap_valence_fd.npz"
MODEL_OUT = ROOT / "models" / "deap_valence_fd.joblib"


def main() -> None:
    if not DATA.exists():
        print(f"Missing {DATA}; run deap_extract_features.py first.")
        return

    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score
    import joblib

    d = np.load(DATA, allow_pickle=True)
    X, y, subjects = d["X"], d["y"], d["subjects"]
    feature_names = list(d["feature_names"])
    print(f"Data: X={X.shape}  high-valence={y.mean():.1%}  subjects={len(np.unique(subjects))}")
    print(f"Features: {feature_names}\n")

    logo = LeaveOneGroupOut()
    accs, baccs, f1s = [], [], []
    # class_weight balanced so the minority (high valence) is not ignored.
    for train_idx, test_idx in logo.split(X, y, groups=subjects):
        clf = make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"),
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        accs.append(accuracy_score(y[test_idx], pred))
        baccs.append(balanced_accuracy_score(y[test_idx], pred))
        f1s.append(f1_score(y[test_idx], pred, zero_division=0))

    print("=== LOSO (leave-one-subject-out) — the honest cross-subject result ===")
    print(f"  raw accuracy      : {np.mean(accs):.1%}  (± {np.std(accs):.1%})")
    print(f"  balanced accuracy : {np.mean(baccs):.1%}  (± {np.std(baccs):.1%})  <-- the real number")
    print(f"  F1 (high valence) : {np.mean(f1s):.3f}")
    chance = max(y.mean(), 1 - y.mean())
    print(f"\n  majority-class baseline (always-predict): {chance:.1%} raw / 50.0% balanced")
    print(f"  literature reference: ~64.9% (Ismail PPG cross-subject), 70.9% DEAP")

    # Final model on ALL subjects, for the Galaxy-Watch transfer test.
    final = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced", probability=True),
    )
    final.fit(X, y)
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": final, "feature_names": feature_names,
                 "valence_split": 5.0, "trained_on": "DEAP"}, MODEL_OUT)
    print(f"\nSaved final model -> {MODEL_OUT}")


if __name__ == "__main__":
    main()
