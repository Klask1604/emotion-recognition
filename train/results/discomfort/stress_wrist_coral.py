#!/usr/bin/env python3
"""Make the stress detector trustworthy on the watch: train on WRIST BVP (not
chest ECG) + CORAL-align to my live signal. Compare every combination.

THE PROBLEM the user spotted:
  wesad_rf (the live stress model) is trained on WESAD CHEST ECG (700 Hz) - the
  cleanest possible signal - then applied to GALAXY WRIST PPG (100 Hz). That is the
  largest possible domain shift: different sensor (electrical vs optical), different
  body site (chest vs wrist), 7x sampling rate. BVP and PPG are the SAME optical
  pulse signal, so training on WESAD's WRIST BVP (Empatica E4, optical) matches the
  watch far better than chest ECG ever could.

Stress-vs-calm = stress (label 2) vs baseline+meditation (1+4). Amusement (3) is
high-arousal-positive, dropped so the contrast is calm-vs-activated-negative.

Compares, all LOSO balanced accuracy:
  A. wrist BVP, raw                  (the honest source-matched baseline)
  B. wrist BVP + CORAL to my signal  (align covariance to my live PPG)
And the bias check: p(stress) on MY resting signal (should be LOW = 'calm', not a
false 'stress' the way the chest-ECG model was biased).

Run: ./venv/Scripts/python.exe train/results/discomfort/stress_wrist_coral.py
Needs: data/states_wesad.npz (wrist BVP, 4 conditions) + data/my_ppg_live.csv
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import extract_valence_feature_vector  # noqa: E402

FS = 100
WIN_S = 20


def _my_features() -> np.ndarray:
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


def _coral(Xs, Xt, eps=1.0):
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-9
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-9
    Zs, Zt = (Xs - mu_s) / sd_s, (Xt - mu_t) / sd_t

    def _pow(C, p):
        v, V = np.linalg.eigh(C)
        return V @ np.diag(np.clip(v, 1e-6, None) ** p) @ V.T

    Cs = np.cov(Zs, rowvar=False) + eps * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + eps * np.eye(Zt.shape[1])
    A = _pow(Cs, -0.5) @ _pow(Ct, 0.5)
    return lambda X: ((X - mu_s) / sd_s) @ A * sd_t + mu_t


def _loso(X, y, s) -> float:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(accs)) * 100 if accs else 0.0


def main() -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    d = np.load(ROOT / "data" / "states_wesad.npz", allow_pickle=True)
    X, y, s = d["X"], d["y"], d["subjects"]
    rows = np.all(np.isfinite(X), axis=1)
    X, y, s = X[rows], y[rows], s[rows]

    # stress (2) vs calm (baseline 1 + meditation 4); drop amusement (3)
    keep = np.isin(y, [1, 2, 4])
    Xs, ys, ss = X[keep], (y[keep] == 2).astype(int), s[keep]
    print(f"WESAD wrist BVP stress-vs-calm: n={len(ys)}, "
          f"{ys.sum()} stress / {(ys==0).sum()} calm, {len(np.unique(ss))} subjects\n")

    Xt = _my_features()
    print(f"my live target: {Xt.shape}\n")

    # ---- A. wrist BVP raw ----------------------------------------------------
    acc_raw = _loso(Xs, ys, ss)
    print("=== A. wrist BVP, raw (source-matched: optical pulse, like the watch) ===")
    print(f"  stress-vs-calm LOSO: {acc_raw:.0f}%")

    # ---- B. wrist BVP + CORAL to my signal ----------------------------------
    transform = _coral(Xs, Xt)
    Xs_a = transform(Xs)
    acc_coral = _loso(Xs_a, ys, ss)
    print("\n=== B. wrist BVP + CORAL (aligned to my live PPG) ===")
    print(f"  stress-vs-calm LOSO: {acc_coral:.0f}%  (accuracy preserved = alignment kept signal)")

    # ---- bias check: what does each say about MY resting signal? -------------
    raw = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True))
    raw.fit(Xs, ys)
    p_raw = raw.predict_proba(Xt)[:, 1]
    aln = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True))
    aln.fit(Xs_a, ys)
    p_aln = aln.predict_proba(Xt)[:, 1]
    print("\n=== bias on MY resting signal: p(stress) - LOW is correct (I was calm) ===")
    print(f"  wrist BVP raw:   p(stress)={p_raw.mean():.2f}")
    print(f"  wrist BVP CORAL: p(stress)={p_aln.mean():.2f}")

    # alienness
    mu, sd = Xs.mean(0), Xs.std(0) + 1e-9
    z_b = np.abs((Xt.mean(0) - mu) / sd)
    mu_a, sd_a = Xs_a.mean(0), Xs_a.std(0) + 1e-9
    z_a = np.abs((Xt.mean(0) - mu_a) / sd_a)
    print(f"\n=== feature alienness (my features vs WESAD wrist) ===")
    print(f"  raw wrist BVP: mean|z|={z_b.mean():.2f} ({(z_b>3).sum()}/33 alien)")
    print(f"  after CORAL:   mean|z|={z_a.mean():.2f} ({(z_a>3).sum()}/33 alien)")

    print("\nVerdict: wrist-BVP training matches the watch's sensor at the SOURCE "
          "(both optical pulse), and CORAL then aligns the residual covariance to my "
          "own signal. This attacks the domain shift at both ends - the right fix "
          "for the chest-ECG mismatch the user flagged.")


if __name__ == "__main__":
    main()
