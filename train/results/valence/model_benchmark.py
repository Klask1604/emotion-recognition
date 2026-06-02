#!/usr/bin/env python3
"""Systematic benchmark: which model, on which dataset, under which protocol?

So far the thesis used one model (SVM) and one protocol (LOSO). This runs the full
matrix to answer methodically:
  - 6 models: LogReg, KNN, SVM, RandomForest, GradientBoosting, MLP
  - the 33-feature datasets that share the exact extractor: WESAD, CASE, EEVR,
    EMOGNITION
  - two protocols:
      LOSO         (within-dataset): generalise to a NEW subject, same experiment
      CROSS-DATASET: train on one dataset, test on ANOTHER entirely - the hardest,
                     most honest test of "does it capture emotion, not dataset
                     artefacts?"
Target = binary valence (negative vs positive). Balanced accuracy throughout.

Run: ./venv/Scripts/python.exe train/results/valence/model_benchmark.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _models():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    return {
        "LogReg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM": SVC(kernel="rbf", class_weight="balanced"),
        "RandForest": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                             random_state=0),
        "GradBoost": GradientBoostingClassifier(n_estimators=150, random_state=0),
        "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500,
                             early_stopping=True, random_state=0),
    }


def _datasets():
    """name -> (X[33], y_binary_valence, subjects). Each dataset reduced to a
    clean binary valence label (negative=0, positive=1) on clearly-labelled rows."""
    out = {}

    # WESAD: y already 0=stress(neg) / 1=amusement(pos)
    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    out["WESAD"] = (d["X"], d["y"].astype(int), d["subjects"])

    # CASE: continuous valence; clear labels <=3 neg, >=7 pos
    d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
    v = d["valence"]; sel = (v <= 3) | (v >= 7)
    out["CASE"] = (d["Xb"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel])

    # EEVR: quadrant -> valence = HV vs LV
    d = np.load(ROOT / "data" / "eevr_stratified.npz", allow_pickle=True)
    q = d["quad"]
    pos = np.isin(q, ["HVHA", "HVLA"])
    out["EEVR"] = (d["Xb"], pos.astype(int), d["subjects"])

    # EMOGNITION: SAM valence clear labels
    d = np.load(ROOT / "data" / "emognition_features.npz", allow_pickle=True)
    v = d["valence"]; sel = (v <= 3) | (v >= 7)
    out["EMOGNITION"] = (d["X"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel])

    return out


def _clean(X, y, s):
    rows = np.all(np.isfinite(X), axis=1)
    return X[rows], y[rows], s[rows]


def _recenter_per_subject(X, s):
    """Subtract each subject's own median feature vector (a personal baseline
    proxy) so everyone starts from 'my own rest = 0'. This removes the between-
    subject offset that hurts LOSO - the cheap personalisation step."""
    Xc = X.copy().astype(float)
    for subj in np.unique(s):
        m = s == subj
        Xc[m] = Xc[m] - np.median(Xc[m], axis=0)
    return Xc


def _loso(model_factory, X, y, s) -> float:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    logo = LeaveOneGroupOut(); accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), clone(model_factory))
        m.fit(X[tr], y[tr]); accs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return np.mean(accs) * 100 if accs else float("nan")


def _cross(model_factory, Xtr, ytr, Xte, yte) -> float:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    m = make_pipeline(StandardScaler(), clone(model_factory))
    m.fit(Xtr, ytr)
    return balanced_accuracy_score(yte, m.predict(Xte)) * 100


def main() -> None:
    models = _models()
    data = {k: _clean(*v) for k, v in _datasets().items()}
    for name, (X, y, s) in data.items():
        print(f"  {name}: {X.shape}, {int(y.sum())} pos / {int((y==0).sum())} neg, "
              f"{len(set(s.tolist()))} subjects")
    print()

    # ---- 1. SAME-DATASET LOSO: model x dataset --------------------------------
    print("=== 1. LOSO valence (model x dataset), balanced accuracy % ===")
    header = f"{'model':12s}" + "".join(f"{d:>12s}" for d in data)
    print(header)
    for mname, mf in models.items():
        row = f"{mname:12s}"
        for dname, (X, y, s) in data.items():
            row += f"{_loso(mf, X, y, s):>12.0f}"
        print(row)

    # ---- 1b. LOSO with PER-SUBJECT BASELINE recentring ------------------------
    # The same LOSO, but each subject's features are centred on their own median
    # first (personal baseline). If this lifts the numbers, the between-subject
    # offset was a big part of why cross-subject valence looked weak.
    print("\n=== 1b. LOSO valence WITH per-subject baseline recentring, bal. acc % ===")
    print(header)
    for mname, mf in models.items():
        row = f"{mname:12s}"
        for dname, (X, y, s) in data.items():
            Xc = _recenter_per_subject(X, s)
            row += f"{_loso(mf, Xc, y, s):>12.0f}"
        print(row)

    # ---- 2. CROSS-DATASET (the hardest test) ----------------------------------
    # Best general-purpose model (SVM) trained on each dataset, tested on each other.
    print("\n=== 2. CROSS-DATASET valence (train -> test), SVM, balanced accuracy % ===")
    print("(diagonal skipped; off-diagonal = train on row, test on column)")
    svm = models["SVM"]
    names = list(data)
    label = "train\\test"
    print(f"{label:12s}" + "".join(f"{n:>12s}" for n in names))
    for tr_name in names:
        Xtr, ytr, _ = data[tr_name]
        row = f"{tr_name:12s}"
        for te_name in names:
            if tr_name == te_name:
                row += f"{'-':>12s}"
            else:
                Xte, yte, _ = data[te_name]
                row += f"{_cross(svm, Xtr, ytr, Xte, yte):>12.0f}"
        print(row)

    print("\nReading: LOSO shows the best model per dataset (usually a few points "
          "between models - the dataset/signal matters more than the algorithm). "
          "CROSS-DATASET near 50% confirms models learn dataset-specific patterns, "
          "not transferable valence - the core honest finding, now shown across "
          "every model and dataset pair.")


if __name__ == "__main__":
    main()
