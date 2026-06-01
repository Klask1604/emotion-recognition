#!/usr/bin/env python3
"""Which features carry DISCOMFORT beyond arousal? (offline, WESAD, no watch)

The decisive test showed: stress-vs-amusement on WESAD = 82% LOSO, but HR alone
(hrv_mean_hr) already reaches 76%, and dropping all HR/arousal-like features still
leaves 81%. So there IS a discomfort signal independent of heart rate (~+5%), and
it lives in pulse MORPHOLOGY. This script pins down exactly which features.

Method (all on the cached stress-vs-amusement features, LOSO balanced accuracy):
  1. HR-residualized Cohen's d: regress each feature on hrv_mean_hr, take the
     residual, measure stress-vs-amusement separation on the residual. A feature
     with large residual-d carries discomfort NOT explained by heart rate.
  2. Add-one-on-top-of-HR: start from HR-only (76%), add each single feature, see
     which lifts accuracy most — the marginal discomfort contribution of each.
  3. Family ablation WITHOUT HR features: which family (vascular/FD/morph) holds
     the non-HR discomfort signal.

Run: ./venv/Scripts/python.exe train/discomfort_features.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import VALENCE_FEATURE_NAMES  # noqa: E402

# HR / arousal-like features (rate-driven) — what we want to look BEYOND.
HR_LIKE = {"hrv_mean_hr", "pulse_width", "hrv_rmssd", "hrv_sdnn", "hrv_pnn50"}
# Family index ranges in the shared extractor order: vascular(6) fd(6) morph(14) hrv(7)
VASC = list(range(0, 6))
FD = list(range(6, 12))
MORPH = list(range(12, 26))
HRV = list(range(26, 33))


def _loso(X, y, s):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score

    logo = LeaveOneGroupOut()
    acc = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        acc.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(acc)) * 100 if acc else 0.0


def main() -> None:
    names = list(VALENCE_FEATURE_NAMES)
    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    X, y, s = d["X"], d["y"], d["subjects"]
    # y: 1=amusement, 0=stress. Make discomfort the positive class for clarity.
    y_disc = 1 - y  # 1 = stress (discomfort), 0 = amusement
    hr = X[:, names.index("hrv_mean_hr")]

    print(f"WESAD discomfort (stress vs amusement) — n={len(y)}, "
          f"{len(np.unique(s))} subjects, discomfort={y_disc.mean():.0%}\n")

    # ---- 1. HR-residualized Cohen's d -----------------------------------------
    print("=== 1. Discomfort separation AFTER removing heart rate (residual d) ===")
    print("    (regress feature on HR, separate stress vs amusement on the residual)")
    hr_c = (hr - hr.mean())
    rows = []
    for i, nm in enumerate(names):
        if nm == "hrv_mean_hr":
            continue
        f = X[:, i].astype(float)
        # residual of f after linear fit on HR
        b = np.polyfit(hr_c, f, 1)
        resid = f - np.polyval(b, hr_c)
        a, c = resid[y_disc == 1], resid[y_disc == 0]
        sd = np.sqrt((a.var() + c.var()) / 2) + 1e-9
        dval = (a.mean() - c.mean()) / sd
        rows.append((nm, dval))
    rows.sort(key=lambda r: -abs(r[1]))
    for nm, dval in rows[:10]:
        fam = ("vasc" if names.index(nm) in VASC else "fd" if names.index(nm) in FD
               else "morph" if names.index(nm) in MORPH else "hrv")
        bar = "#" * int(min(abs(dval) * 12, 28))
        print(f"  {nm:18s} [{fam:5s}] d={dval:+.2f}  {bar}")

    # ---- 2. add-one-on-top-of-HR ----------------------------------------------
    print("\n=== 2. Marginal lift: HR-only (76%) + each single feature ===")
    hr_idx = [names.index("hrv_mean_hr")]
    base = _loso(X[:, hr_idx], y_disc, s)
    print(f"  HR-only baseline: {base:.0f}%")
    lifts = []
    for i, nm in enumerate(names):
        if nm == "hrv_mean_hr":
            continue
        acc = _loso(X[:, hr_idx + [i]], y_disc, s)
        lifts.append((nm, acc - base))
    lifts.sort(key=lambda r: -r[1])
    for nm, lift in lifts[:8]:
        fam = ("vasc" if names.index(nm) in VASC else "fd" if names.index(nm) in FD
               else "morph" if names.index(nm) in MORPH else "hrv")
        print(f"  +{nm:18s} [{fam:5s}] -> {base + lift:.0f}%  ({lift:+.1f} pp)")

    # ---- 3. family ablation without HR ----------------------------------------
    print("\n=== 3. Which family holds the non-HR discomfort signal ===")
    non_hr = [i for i, n in enumerate(names) if n not in HR_LIKE]
    print(f"  all 33                : {_loso(X, y_disc, s):.0f}%")
    print(f"  non-HR (no rate feat) : {_loso(X[:, non_hr], y_disc, s):.0f}%")
    for fam_name, idx in (("vascular", VASC), ("freq-domain", FD),
                          ("morphology", MORPH), ("hrv-spectral", [26+i for i in range(7) if names[26+i] not in HR_LIKE])):
        sub = [i for i in idx if names[i] not in HR_LIKE]
        if sub:
            print(f"  {fam_name:14s} only   : {_loso(X[:, sub], y_disc, s):.0f}%  ({len(sub)} feat)")

    print("\nReading: features with high residual-d (1) and high marginal lift (2) are "
          "the discomfort signal BEYOND heart rate — the morphological core to validate "
          "on the watch. If they're all 'morph', discomfort rides on pulse SHAPE, which "
          "is exactly what domain-shifts on the wrist (so expect it to weaken live).")


if __name__ == "__main__":
    main()
