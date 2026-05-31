#!/usr/bin/env python3
"""How far is MY real 100 Hz watch signal from what the models learned?

The domain-shift number (AUC=1.000) was measured on GalaxyPPG at 25 Hz. But the
live watch now runs the valence models at 100 Hz (on-demand PPG). This script
measures the shift on MY OWN 100 Hz signal: extract the 33 features from an hour
of live watch PPG (data/my_ppg_live.csv), and compare their distribution to the
WESAD training features (data/wesad_features.npz) the models were trained on.

If a classifier can still tell "mine" from "WESAD" perfectly, the shift is as bad
at 100 Hz as at 25 Hz. If it drops, the better sample rate helped.

Run:  ./venv/Scripts/python.exe train/my_domain_shift.py
Needs: data/my_ppg_live.csv  (exported from Influx), data/wesad_features.npz
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

WIN_S, STEP_S = 20, 10
FS = 100


def extract_mine() -> np.ndarray:
    """Window my live PPG into the same 33-feature vectors the models consume."""
    df = pd.read_csv(ROOT / "data" / "my_ppg_live.csv")
    df = df[df["green"].notna()]
    # parse time to ms epoch
    t = pd.to_datetime(df["time"], utc=True, format="ISO8601").astype("int64") // 1_000_000  # ms
    g = df["green"].to_numpy(float)
    t = t.to_numpy()
    X = []
    w, st = WIN_S * FS, STEP_S * FS
    for i in range(0, len(g) - w + 1, st):
        seg_g = [int(round(x)) for x in g[i:i + w]]
        seg_t = [int(x) for x in t[i:i + w]]
        # need a monotonic ~20 s window (skip gaps where the watch paused)
        span = (seg_t[-1] - seg_t[0]) / 1000.0
        if not (15 <= span <= 30):
            continue
        vec = extract_valence_feature_vector(seg_g, seg_t)
        if vec is not None and all(np.isfinite(vec)):
            X.append(vec)
    return np.asarray(X, float)


def main() -> None:
    from sklearn.model_selection import cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    names = list(VALENCE_FEATURE_NAMES)
    print("Extracting 33 features from MY live 100 Hz PPG...")
    Xmine = extract_mine()
    print(f"  {len(Xmine)} windows from my watch")

    wes = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    Xwes = wes["X"]
    print(f"  {len(Xwes)} windows from WESAD training (Empatica E4, 64 Hz)")
    if len(Xmine) < 10:
        print("Too few windows from my PPG (gaps?). Need a longer clean stretch.")
        return

    # ---- domain classifier: can it tell mine from WESAD? -----------------------
    X = np.vstack([Xmine, Xwes])
    y = np.r_[np.ones(len(Xmine)), np.zeros(len(Xwes))]   # 1=mine, 0=WESAD
    rows = np.all(np.isfinite(X), axis=1)
    X, y = X[rows], y[rows]
    clf = make_pipeline(StandardScaler(),
                        SVC(kernel="rbf", probability=True, class_weight="balanced"))
    auc = cross_val_score(clf, X, y, cv=5, scoring="roc_auc").mean()
    print(f"\n=== Domain separability (mine vs WESAD), MY real 100 Hz ===")
    print(f"  distinguish-score: {auc:.3f}  "
          f"({'PERFECT shift — as bad as 25 Hz' if auc > 0.97 else 'large' if auc > 0.85 else 'moderate' if auc > 0.7 else 'small — 100 Hz helped!'})")
    print(f"  (1.0 = my signal looks totally alien to the models; 0.5 = looks the same)")

    # ---- which features differ most (so we know what to drop / trust) ----------
    mu_m, mu_w = Xmine.mean(0), Xwes.mean(0)
    sd = np.sqrt((Xmine.var(0) + Xwes.var(0)) / 2) + 1e-9
    d = np.abs((mu_m - mu_w) / sd)
    print(f"\n=== Which features differ most between my watch and WESAD ===")
    print(f"  (high = that feature is in a totally different range on my watch)")
    for i in np.argsort(-d)[:10]:
        print(f"  {names[i]:18s} |d|={d[i]:.1f}")
    print(f"\n  features that MATCH well (|d|<0.5, safe to trust on the watch):")
    safe = [names[i] for i in range(len(names)) if d[i] < 0.5]
    print("   ", ", ".join(safe) if safe else "(none — every feature shifted)")

    print("\nReading: features with small |d| transfer (the models see them in a "
          "familiar range); large |d| are the shift. If RMSSD/HRV-rate features are "
          "in the safe list, the discomfort signal that rides on them survives at 100 Hz.")


if __name__ == "__main__":
    main()
