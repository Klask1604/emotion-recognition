#!/usr/bin/env python3
"""
Train the WESAD valence model used by the production valence fusion channel.

Valence axis = stress (label 2, negative) vs amusement (label 3, positive) —
two high-arousal states that differ only in valence. Features: PPG vascular-tone
(perfusion index etc.) + frequency-domain harmonics + pulse morphology + HRV,
the same extractors the live channel uses (train/serve parity). Wrist BVP
(Empatica E4, 64 Hz) so it matches the watch's optical PPG, not chest ECG.

Reports LOSO balanced accuracy (the honest cross-subject number, ~85% with
vascular features) and saves models/valence_wesad.joblib as a bundle
{model, feature_names, feature_mean, feature_std, classes}.

Usage:
    ./venv/Scripts/python.exe train/train_valence_wesad.py
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from affectus.research.valence.features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 64
WIN_S = 20
STEP_S = 10
LABEL_STRESS, LABEL_AMUSEMENT = 2, 3
OUT = ROOT / "models" / "valence_wesad.joblib"


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score
    import joblib

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
            dom = vals[np.argmax(cnts)]
            if dom not in (LABEL_STRESS, LABEL_AMUSEMENT):
                continue
            vec = extract_valence_feature_vector(
                [int(round(x)) for x in seg],
                [int(j * 1000 / FS) for j in range(len(seg))],
            )
            if vec is None:
                continue
            X.append(vec)
            y.append(1 if dom == LABEL_AMUSEMENT else 0)  # 1 = positive valence
            subj.append(sid)

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subj = np.asarray(subj, dtype=int)
    print(f"n={len(y)}, subjects={len(np.unique(subj))}, positive(amusement)={y.mean():.1%}")

    # Honest LOSO score (report only).
    logo = LeaveOneGroupOut()
    baccs = []
    for tr, te in logo.split(X, y, groups=subj):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        baccs.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    print(f"LOSO balanced accuracy: {np.mean(baccs):.1%} (± {np.std(baccs):.1%})")

    # Final model on all subjects, with probability for the fusion channel.
    mean = X.mean(axis=0)
    std = X.std(axis=0) + 1e-9
    Xn = (X - mean) / std
    final = LogisticRegression(max_iter=1000, class_weight="balanced")
    final.fit(Xn, y)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({
        "model": final,
        "feature_names": VALENCE_FEATURE_NAMES,
        "feature_mean": mean,
        "feature_std": std,
        "classes": ["negative", "positive"],
        "trained_on": "WESAD stress-vs-amusement (valence axis)",
    }, OUT)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
