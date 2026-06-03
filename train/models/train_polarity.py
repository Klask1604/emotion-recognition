#!/usr/bin/env python3
"""Train the polarity detector (negative / neutral / positive) for the live watch.

The product question: is the user feeling something NEGATIVE, POSITIVE, or nothing
in particular (NEUTRAL) right now? A two-class stress-vs-amusement model has no way
to represent rest - it forces every resting window into one extreme. So we train
THREE classes:
    NEGATIVE = WESAD stress (2)
    POSITIVE = WESAD amusement (3)
    NEUTRAL  = WESAD baseline (1) + meditation (4)
This gives the model a real 'rest' class, so at rest it says NEUTRAL instead of a
false NEGATIVE. LOSO: negative recall ~84% (strong), neutral ~57%, positive ~36%
(positive is the hardest, the smallest class - the model stays conservative and
calls a weak positive 'neutral' rather than inventing it). Neg<->pos confusion is
low (<13%), so when it asserts a sign, the sign is rarely wrong - the honest
behaviour we want for real discomfort detection.

Two domain-shift fixes baked into the bundle, the same that made the stress
detector trustworthy on my real signal:
  1. WRIST BVP source (Empatica E4, optical) - same sensor family as the watch PPG,
     not chest ECG. Extracted in states_wesad.npz.
  2. CORAL alignment - the saved transform maps WESAD-feature covariance onto my
     live PPG, so the model stops asserting false confidence on my signal.

Bundle (models/polarity.joblib) matches the existing valence bundle shape plus a
`coral_transform` matrix the engine applies before normalisation:
  model, feature_names, feature_mean, feature_std, classes, trained_on,
  coral_A, coral_mu_s, coral_sd_s, coral_sd_t   (the CORAL transform pieces)

Run: ./venv/Scripts/python.exe train/models/train_polarity.py
Needs: data/states_wesad.npz (wrist BVP, 4 conditions) + data/my_ppg_live.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 100
WIN_S = 20
OUT = ROOT / "models" / "polarity.joblib"


def _my_features() -> np.ndarray:
    """My live PPG features - the CORAL target distribution."""
    df = pd.read_csv(ROOT / "data" / "my_ppg_live.csv")
    df = df[df["green"].notna()]
    g = df["green"].to_numpy(float)
    t = pd.to_datetime(df["time"], utc=True, format="ISO8601").astype("int64") // 1_000_000
    t = t.to_numpy()
    w = FS * WIN_S
    vecs = []
    for i in range(0, len(g) - w + 1, w // 2):
        seg_g = [int(round(x)) for x in g[i:i + w]]
        seg_t = [int(x) for x in t[i:i + w]]
        if 15 <= (seg_t[-1] - seg_t[0]) / 1000.0 <= 30:
            vec = extract_valence_feature_vector(seg_g, seg_t)
            if vec is not None and all(np.isfinite(vec)):
                vecs.append(vec)
    return np.asarray(vecs, float)


def _coral_pieces(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0):
    """Return the CORAL transform pieces so the engine can rebuild it at runtime."""
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-9
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-9
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t

    def _pow(C, p):
        v, V = np.linalg.eigh(C)
        return V @ np.diag(np.clip(v, 1e-6, None) ** p) @ V.T

    Cs = np.cov(Zs, rowvar=False) + eps * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + eps * np.eye(Zt.shape[1])
    A = _pow(Cs, -0.5) @ _pow(Ct, 0.5)
    return A, mu_s, sd_s, sd_t


def _apply_coral(X, A, mu_s, sd_s, sd_t):
    return ((X - mu_s) / sd_s) @ A * sd_t + mu_s


def main() -> None:
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.metrics import balanced_accuracy_score

    d = np.load(ROOT / "data" / "states_wesad.npz", allow_pickle=True)
    X, y4, s = d["X"], d["y"], d["subjects"]
    rows = np.all(np.isfinite(X), axis=1)
    X, y4, s = X[rows], y4[rows], s[rows]

    # 3 classes: 0=neutral (baseline 1 + meditation 4), 1=negative (stress 2),
    # 2=positive (amusement 3).
    y = np.full(len(y4), -1)
    y[np.isin(y4, [1, 4])] = 0
    y[y4 == 2] = 1
    y[y4 == 3] = 2
    keep = y >= 0
    X, y, s = X[keep], y[keep], s[keep]
    print(f"polarity 3-class: n={len(y)}, "
          f"neutral={int((y==0).sum())} negative={int((y==1).sum())} "
          f"positive={int((y==2).sum())}, {len(np.unique(s))} subjects")

    Xt = _my_features()
    print(f"CORAL target (my live PPG): {Xt.shape}")
    A, mu_s, sd_s, sd_t = _coral_pieces(X, Xt)
    Xc = _apply_coral(X, A, mu_s, sd_s, sd_t)

    # LOSO sanity + per-class recall (report; the bundle trains on all)
    logo = LeaveOneGroupOut()
    yt, yp = [], []
    for tr, te in logo.split(Xc, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        fm, fs = Xc[tr].mean(0), Xc[tr].std(0) + 1e-9
        m = LogisticRegression(max_iter=1000, class_weight="balanced")
        m.fit((Xc[tr] - fm) / fs, y[tr])
        yt += list(y[te]); yp += list(m.predict((Xc[te] - fm) / fs))
    yt, yp = np.array(yt), np.array(yp)
    print(f"LOSO balanced accuracy (3-class, wrist BVP + CORAL): "
          f"{balanced_accuracy_score(yt, yp) * 100:.0f}%")
    for i, nm in enumerate(["neutral", "negative", "positive"]):
        mask = yt == i
        if mask.sum():
            print(f"  {nm:9s} recall: {(yp[mask] == i).mean() * 100:.0f}%")

    # final model on all CORAL-aligned data
    feature_mean = Xc.mean(0)
    feature_std = Xc.std(0) + 1e-9
    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit((Xc - feature_mean) / feature_std, y)

    bundle = {
        "model": model,
        "feature_names": list(VALENCE_FEATURE_NAMES),
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "classes": ["neutral", "negative", "positive"],   # class index 0/1/2
        "trained_on": "WESAD wrist BVP neg/neutral/pos (stress / baseline+meditation / amusement) + CORAL",
        # CORAL transform pieces - engine applies before normalisation
        "coral_A": A,
        "coral_mu_s": mu_s,
        "coral_sd_s": sd_s,
        "coral_sd_t": sd_t,
    }
    import joblib
    OUT.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, OUT)
    print(f"\nSaved: {OUT}")
    print("Bundle carries the CORAL transform so the live engine aligns each "
          "feature window to the watch's own distribution before predicting.")


if __name__ == "__main__":
    main()
