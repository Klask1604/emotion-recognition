#!/usr/bin/env python3
"""Native-state classifier on WESAD — distinguish the ACTUAL conditions, not valence.

The valence models collapse the rich WESAD conditions into a weak binary axis
(negative vs positive, ~58% — the valence ceiling). But WESAD has FOUR real states
the watch could plausibly report: baseline, stress, amusement, meditation. Three of
them differ in AROUSAL (which is strong), so a model that predicts the STATE
directly — instead of forcing the Russell valence axis — should do better.

This is a SEPARATE model. It does NOT touch the valence models or the Grafana
boards (those keep running). It just answers the user's question: "does using the
models' native states work better than the forced 2D valence axis?"

Two views, LOSO balanced accuracy:
  1. 4-state: baseline / stress / amusement / meditation (the full native task).
  2. stress-vs-calm: stress vs (baseline+meditation) — the useful product target
     (am I stressed or not), where arousal does the heavy lifting.

Run: ./venv/Scripts/python.exe train/states_wesad.py
Cache: data/states_wesad.npz
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 64
WIN_S = 20
COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
CACHE = ROOT / "data" / "states_wesad.npz"


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
            if dom not in COND:
                continue
            vec = extract_valence_feature_vector(
                [int(round(x)) for x in seg],
                [int(j * 1000 / FS) for j in range(len(seg))],
            )
            if vec is None:
                continue
            X.append(vec); y.append(dom); subj.append(sid)
    out = dict(X=np.asarray(X, float), y=np.asarray(y, int), subjects=np.asarray(subj, int))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    return out


def _loso(X, y, s, classes_min=2):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score, confusion_matrix

    logo = LeaveOneGroupOut()
    accs, yt_all, yp_all = [], [], []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < classes_min:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        pred = c.predict(X[te])
        accs.append(balanced_accuracy_score(y[te], pred))
        yt_all += list(y[te]); yp_all += list(pred)
    return float(np.mean(accs)) * 100 if accs else 0.0, np.array(yt_all), np.array(yp_all)


def main() -> None:
    d = extract()
    X, y, s = d["X"], d["y"], d["subjects"]
    from collections import Counter
    print(f"WESAD native states — n={len(y)}, {len(np.unique(s))} subjects")
    print("per state:", {COND[k]: v for k, v in sorted(Counter(y).items())}, "\n")

    # ---- 1. full 4-state ------------------------------------------------------
    print("=== 1. 4-state (baseline/stress/amusement/meditation), LOSO ===")
    acc4, yt, yp = _loso(X, y, s)
    print(f"  balanced accuracy: {acc4:.0f}%  (chance 25%)")
    print("  per-state recall:")
    for c in sorted(COND):
        m = yt == c
        if m.sum():
            print(f"    {COND[c]:11s}: {(yp[m]==c).mean()*100:4.0f}%  (n={m.sum()})")

    # ---- 2. stress vs calm (the useful product target) ------------------------
    print("\n=== 2. STRESS vs CALM (stress vs baseline+meditation), LOSO ===")
    keep = np.isin(y, [1, 2, 4])
    Xs, ys, ss = X[keep], y[keep], s[keep]
    yb = (ys == 2).astype(int)   # 1 = stress, 0 = calm
    accs, _, _ = _loso(Xs, yb, ss)
    print(f"  balanced accuracy: {accs:.0f}%  (chance 50%)")
    print("  -> this is 'am I stressed or not' — arousal carries it, should be high")

    # ---- 3. compare to the valence binary (the weak axis) --------------------
    print("\n=== 3. Reference: valence binary (stress vs amusement) ===")
    keep2 = np.isin(y, [2, 3])
    yv = (y[keep2] == 3).astype(int)
    accv, _, _ = _loso(X[keep2], yv, s[keep2])
    print(f"  balanced accuracy: {accv:.0f}%  (the 'valence' number)")

    print("\nVerdict: if STRESS-vs-CALM (2) is well above the valence binary (3), then "
          "reporting NATIVE STATES (stressed / calm / engaged) works better than forcing "
          "the weak Russell valence axis — the user's intuition. States ride on arousal.")


if __name__ == "__main__":
    main()
