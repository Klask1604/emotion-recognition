#!/usr/bin/env python3
"""CORAL domain adaptation: can we align WESAD features to MY watch and reduce
the domain shift that makes WESAD predict 'stress' on my resting signal?

my_domain_shift.py proved the raw features are far from WESAD (morphology shifted
beyond use). Personal-baseline recentring already removes the MEAN bias. CORAL goes
further: it matches the COVARIANCE structure of the source (WESAD) to the target
(my live PPG), an unsupervised second-order alignment - no target labels needed.

  CORAL (Sun & Saenko 2016): whiten source by its covariance, recolour by target's.
      Cs = cov(source) + eps*I ;  Ct = cov(target) + eps*I
      A  = Cs^-1/2 @ Ct^1/2
      source_aligned = (source - mu_s) @ A + mu_t

Test: does a stress-vs-amusement classifier trained on CORAL-aligned WESAD assign
LESS extreme/biased valence to my resting signal than the raw model? We can't
measure accuracy on me (no labels), so we measure:
  1. WESAD LOSO accuracy is PRESERVED after alignment (we didn't break the model).
  2. On MY features, the aligned model's p_positive sits closer to 0.5 (undecided
     on resting) instead of the raw model's stuck-low ~0.25 (false 'stress').

Run: ./venv/Scripts/python.exe train/results/discomfort/coral_domain_adapt.py
Needs: data/wesad_features.npz + data/my_ppg_live.csv
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
    """Extract the 33-feature vectors from an hour of my live watch PPG."""
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
        span = (seg_t[-1] - seg_t[0]) / 1000.0
        if 15 <= span <= 30:
            vec = extract_valence_feature_vector(seg_g, seg_t)
            if vec is not None and all(np.isfinite(vec)):
                vecs.append(vec)
    return np.asarray(vecs, float)


def _coral(Xs: np.ndarray, Xt: np.ndarray, eps: float = 1.0):
    """Return a transform that maps source features into the target's covariance.
    Standardise both first so scale differences don't dominate, then match cov."""
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-9
    mu_t, sd_t = Xt.mean(0), Xt.std(0) + 1e-9
    Zs = (Xs - mu_s) / sd_s
    Zt = (Xt - mu_t) / sd_t

    def _mat_pow(C, p):
        vals, vecs = np.linalg.eigh(C)
        vals = np.clip(vals, 1e-6, None)
        return vecs @ np.diag(vals ** p) @ vecs.T

    Cs = np.cov(Zs, rowvar=False) + eps * np.eye(Zs.shape[1])
    Ct = np.cov(Zt, rowvar=False) + eps * np.eye(Zt.shape[1])
    A = _mat_pow(Cs, -0.5) @ _mat_pow(Ct, 0.5)

    def transform(X):
        Z = (X - mu_s) / sd_s
        Zc = Z @ A
        return Zc * sd_t + mu_t        # land in target's standardised frame, de-std by target

    return transform


def _loso(X, y, s) -> float:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True))
        c.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(accs)) * 100 if accs else 0.0


def main() -> None:
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    Xs, ys, ss = d["X"], d["y"].astype(int), d["subjects"]
    rows = np.all(np.isfinite(Xs), axis=1)
    Xs, ys, ss = Xs[rows], ys[rows], ss[rows]

    print("Extracting my live features...")
    Xt = _my_features()
    print(f"  WESAD source: {Xs.shape} | my target: {Xt.shape}\n")
    if len(Xt) < 20:
        print("Too few clean windows from my signal - aborting.")
        return

    # ---- 1. does alignment PRESERVE WESAD's own accuracy? --------------------
    acc_raw = _loso(Xs, ys, ss)
    transform = _coral(Xs, Xt)
    Xs_aligned = transform(Xs)
    acc_aligned = _loso(Xs_aligned, ys, ss)
    print("=== 1. WESAD LOSO accuracy (did we break the model?) ===")
    print(f"  raw:     {acc_raw:.0f}%")
    print(f"  CORAL:   {acc_aligned:.0f}%  (should stay similar - alignment kept the signal)\n")

    # ---- 2. what does each model say about MY resting signal? ----------------
    # Train on full WESAD (raw vs aligned), predict p_positive on my features.
    raw_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True))
    raw_clf.fit(Xs, ys)
    p_raw = raw_clf.predict_proba(Xt)[:, 1]

    aln_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced", probability=True))
    aln_clf.fit(Xs_aligned, ys)
    # my features must enter the SAME aligned frame the classifier learned -> identity
    # for target (CORAL maps source->target, target stays itself), so predict directly.
    p_aln = aln_clf.predict_proba(Xt)[:, 1]

    print("=== 2. p_positive on MY resting signal (0.5 = undecided, <0.5 = false 'negative/stress') ===")
    print(f"  raw model:   mean={p_raw.mean():.2f}  (stuck low = domain-shift bias toward 'stress')")
    print(f"  CORAL model: mean={p_aln.mean():.2f}  (closer to 0.5 = less false confidence)")
    print(f"  shift toward neutral: {abs(p_aln.mean()-0.5) < abs(p_raw.mean()-0.5)}")

    # how 'alien' are my features before vs after aligning the source to me?
    mu_s, sd_s = Xs.mean(0), Xs.std(0) + 1e-9
    z_before = np.abs((Xt.mean(0) - mu_s) / sd_s)
    mu_a = Xs_aligned.mean(0); sd_a = Xs_aligned.std(0) + 1e-9
    z_after = np.abs((Xt.mean(0) - mu_a) / sd_a)
    print(f"\n=== 3. feature alienness (mean |z| of my features vs WESAD) ===")
    print(f"  before CORAL: {z_before.mean():.2f}  (features >3 = {(z_before>3).sum()}/33)")
    print(f"  after  CORAL: {z_after.mean():.2f}  (features >3 = {(z_after>3).sum()}/33)")

    print("\nReading: CORAL helps IF (1) WESAD accuracy is preserved AND (2) my "
          "resting p_positive moves toward 0.5 (model stops asserting false stress) "
          "AND (3) feature alienness drops. If accuracy collapses, the morphology "
          "shift is too deep for second-order alignment - confirming the physical limit.")


if __name__ == "__main__":
    main()
