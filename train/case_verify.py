#!/usr/bin/env python3
"""Is the CASE 63% arousal-matched valence real, or an artifact?

case_valence_stratified.py found HV-vs-LV at HIGH arousal (matched) = 63.1% on the
SHARED extractor (p=0.014) but only 49.2% on the FAMILY extractor. Before building
anything on 63%, verify three things on the cached features (no re-extraction):

  1. SEED STABILITY: the matching subsamples randomly (seed=0). Re-run the whole
     HIGH-band matched LOSO over many seeds — report mean±std. If 63% was a lucky
     subsample, the mean drops and std is large.
  2. FAMILY vs SHARED: the only difference is normalize_ppg_window (median+MAD) in
     shared. Re-run both over the SAME matched sets per seed so the comparison is
     apples-to-apples (not different random subsets).
  3. WHICH FEATURES: permutation importance on the shared extractor in the HIGH
     band — is the signal in a few sane features (vascular/morph) or smeared/leaky?

Run: ./venv/Scripts/python.exe train/case_verify.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.case_valence_stratified import (  # noqa: E402
    HV_THR, LV_THR, HI_AROUSAL, _match_arousal,
)
from train.eevr_valence_stratified import loso, FAMILY_VASC_IDX, FAMILY_NONVASC_IDX  # noqa: E402
from affectus.legacy.valence_features import VALENCE_FEATURE_NAMES  # noqa: E402

CACHE = ROOT / "data" / "case_stratified.npz"


def high_band_matched(d, seed):
    """Return (Xa, Xb, y, g) for HV-vs-LV in the high-arousal band, arousal-matched."""
    val, aro, subj = d["valence"], d["arousal"], d["subjects"]
    band = aro >= HI_AROUSAL
    hv, lv = val >= HV_THR, val <= LV_THR
    sel = band & (hv | lv)
    y = hv[sel].astype(int)
    a = aro[sel]; g = subj[sel]
    Xa, Xb = d["Xa"][sel], d["Xb"][sel]
    keep = _match_arousal(a, y, np.random.default_rng(seed))
    return Xa[keep], Xb[keep], y[keep], g[keep], a[keep]


def main():
    d = np.load(CACHE, allow_pickle=True)
    d = {k: d[k] for k in d.files}

    # ---- 1 + 2: seed stability, family vs shared on the SAME matched sets ----
    print("=== Seed stability (HIGH arousal, matched) — 15 seeds ===")
    fam, sha, aps = [], [], []
    for s in range(15):
        Xa, Xb, y, g, a = high_band_matched(d, s)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        _, p = ttest_ind(a[y == 1], a[y == 0], equal_var=False)
        fa, _ = loso(Xa, y, g)
        fs, _ = loso(Xb, y, g)
        fam.append(fa); sha.append(fs); aps.append(p)
    fam, sha = np.array(fam), np.array(sha)
    print(f"  n_seeds={len(fam)}, mean windows~{len(y)}, arousal-match p mean={np.mean(aps):.2f}")
    print(f"  FAMILY : {fam.mean():5.1f}% ± {fam.std():.1f}   (range {fam.min():.0f}-{fam.max():.0f})")
    print(f"  SHARED : {sha.mean():5.1f}% ± {sha.std():.1f}   (range {sha.min():.0f}-{sha.max():.0f})")
    print(f"  -> shared {'STABLE above chance' if sha.mean()-sha.std() > 52 else 'FRAGILE / near chance'}")

    # ablations on shared, averaged over seeds (shared uses VALENCE_FEATURE_NAMES order:
    # vascular, fd, morph, hrv — vascular is the FIRST 6)
    n_vasc = 6
    vasc_idx = list(range(n_vasc))
    rest_idx = list(range(n_vasc, len(VALENCE_FEATURE_NAMES)))
    sv, sr = [], []
    for s in range(15):
        Xa, Xb, y, g, a = high_band_matched(d, s)
        if y.sum() < 5 or (1 - y).sum() < 5:
            continue
        v, _ = loso(Xb[:, vasc_idx], y, g)
        r, _ = loso(Xb[:, rest_idx], y, g)
        sv.append(v); sr.append(r)
    print(f"  SHARED vascular-only : {np.mean(sv):.1f}% ± {np.std(sv):.1f}")
    print(f"  SHARED fd+morph+hrv  : {np.mean(sr):.1f}% ± {np.std(sr):.1f}")

    # ---- 3: which features carry it (permutation importance on shared, seed 0) ----
    print("\n=== Which features drive the shared signal (perm. importance) ===")
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    Xa, Xb, y, g, a = high_band_matched(d, 0)
    names = list(VALENCE_FEATURE_NAMES)
    logo = LeaveOneGroupOut()
    imp = np.zeros(len(names)); nf = 0
    for tr, te in logo.split(Xb, y, groups=g):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(Xb[tr], y[tr])
        r = permutation_importance(clf, Xb[te], y[te], n_repeats=8,
                                   scoring="balanced_accuracy", random_state=0)
        imp += r.importances_mean; nf += 1
    imp /= max(nf, 1)
    for i in np.argsort(-imp)[:10]:
        bar = "#" * int(min(max(imp[i], 0) * 200, 25))
        flag = " <- arousal-proxy?" if names[i] in ("hrv_mean_hr", "pulse_width") else ""
        print(f"  {names[i]:18s} {imp[i]:+.4f}  {bar}{flag}")

    print("\nVerdict: if SHARED mean-std > 52% AND top features are vascular/morph "
          "(not hrv_mean_hr/pulse_width), the 63% is a real—if modest—valence signal.")


if __name__ == "__main__":
    main()
