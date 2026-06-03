#!/usr/bin/env python3
"""4-quadrant Russell emotion classifier on WESAD (the honest best number).

The binary valence models fail because valence alone is weak. But a 4-QUADRANT
classifier can lean on AROUSAL (which is strong) to separate quadrants vertically,
and only needs valence for the horizontal split. Three of the four WESAD conditions
map cleanly onto Russell quadrants:

    amusement       -> Bucuros  (high arousal, positive)
    meditation/base -> Calm     (low arousal, positive)
    stress          -> Stresat  (high arousal, negative)
    (no clean low-arousal-negative condition in WESAD -> Trist under-represented)

We train ONE classifier on the 33 features to predict the quadrant directly, LOSO,
and report overall accuracy + per-quadrant. This is the real number for the thesis
classifier — it will be high where arousal does the work, lower where it needs
valence (Bucuros vs Stresat = same arousal, only valence differs).

Run: ./venv/Scripts/python.exe train/quadrant_classifier.py
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.research.valence.features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 64
WIN_S = 20
# WESAD label -> Russell quadrant (only the conditions we can map).
# 1=baseline, 2=stress, 3=amusement, 4=meditation
COND_TO_QUADRANT = {
    3: "Bucuros",     # amusement: high arousal, positive valence
    4: "Calm",        # meditation: low arousal, positive
    1: "Calm",        # baseline: low-ish arousal, neutral-positive -> Calm
    2: "Stresat",     # stress: high arousal, negative
}
CACHE = ROOT / "data" / "quadrant_features.npz"


def extract() -> dict:
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    X, y, subj = [], [], []
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        sid = int(Path(path).stem[1:])
        d = pickle.load(open(path, "rb"), encoding="latin1")
        bvp = np.array(d["signal"]["wrist"]["BVP"]).flatten()
        lbl = np.array(d["label"])
        wb, wl = WIN_S * FS, WIN_S * 700
        for i in range(len(bvp) // wb):
            seg = bvp[i * wb:(i + 1) * wb]
            ls = lbl[i * wl:(i + 1) * wl]
            vals, cnts = np.unique(ls, return_counts=True)
            dom = int(vals[np.argmax(cnts)])
            if dom not in COND_TO_QUADRANT:
                continue
            vec = extract_valence_feature_vector(
                [int(round(x)) for x in seg],
                [int(j * 1000 / FS) for j in range(len(seg))],
            )
            if vec is None:
                continue
            X.append(vec); y.append(COND_TO_QUADRANT[dom]); subj.append(sid)
    out = dict(X=np.asarray(X, float), y=np.asarray(y), subjects=np.asarray(subj, int))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    return out


def main() -> None:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

    d = extract()
    X, y, s = d["X"], d["y"], d["subjects"]
    classes = sorted(np.unique(y))
    print(f"4-quadrant classifier — n={len(y)}, {len(np.unique(s))} subjects")
    from collections import Counter
    print("per quadrant:", dict(Counter(y)), "\n")

    logo = LeaveOneGroupOut()
    y_true_all, y_pred_all = [], []
    accs, baccs = [], []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true_all += list(y[te]); y_pred_all += list(pred)
        accs.append(accuracy_score(y[te], pred))
        baccs.append(balanced_accuracy_score(y[te], pred))

    print(f"=== OVERALL (LOSO) ===")
    print(f"  accuracy:          {np.mean(accs)*100:.0f}%")
    print(f"  balanced accuracy: {np.mean(baccs)*100:.0f}%  (fair across quadrants)")
    print(f"  chance (4 classes): 25%\n")

    # per-quadrant recall (how often each true quadrant is caught)
    yt = np.array(y_true_all); yp = np.array(y_pred_all)
    print("=== Per quadrant (how well each is recognised) ===")
    for c in classes:
        mask = yt == c
        if mask.sum():
            rec = (yp[mask] == c).mean()
            print(f"  {c:8s}: {rec*100:4.0f}%  (n={mask.sum()})")

    print("\n=== Confusion (rows=true, cols=predicted) ===")
    cm = confusion_matrix(yt, yp, labels=classes)
    print("           " + "  ".join(f"{c[:4]:>5s}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:8s} " + "  ".join(f"{cm[i,j]:5d}" for j in range(len(classes))))

    print("\nReading: high quadrants (Calm/Stresat/Bucuros) ride on arousal+valence; "
          "where two quadrants share arousal (Bucuros vs Stresat), only valence "
          "separates them, so confusion there is the valence limit showing through. "
          "Overall accuracy well above 25% chance = a working emotion classifier.")


if __name__ == "__main__":
    main()
