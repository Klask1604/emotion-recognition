#!/usr/bin/env python3
"""CASE valence, AROUSAL-MATCHED — the cleanest arousal-isolation test.

CASE is the best dataset for the valence question because it has CONTINUOUS
valence AND arousal (joystick ~20Hz, scale 0.5-9.5) with LIVED labels. Unlike
EEVR (coarse quadrant labels, design-truth), here we can match arousal PROPERLY:
restrict to a narrow arousal band, then classify HV vs LV inside it.

This directly tests the WESAD interpretability finding (wesad_explain.py): the
WESAD model leans on hrv_mean_hr (= arousal) and pulse_width (HR-correlated), not
on true valence. If valence is real, HV-vs-LV should beat chance EVEN with arousal
held constant. If it's just arousal, it collapses to chance in-band — exactly like
EEVR. The plain-valence baseline earlier got ~60-63% on CASE; how much was arousal?

Reuses: case_extract.py loader pattern, eevr_valence_stratified feats_family/feats_shared/loso.

Experiments (LOSO, 30 subjects, balanced accuracy):
  1. HV vs LV in HIGH arousal band (arousal >= 6)  -> PRIMARY, mirror of HVHA-vs-LVHA
  2. HV vs LV in LOW  arousal band (arousal <= 4)
  3. HV vs LV, no arousal control (plain-valence baseline)  -> delta(3-1) = arousal's share
Valence thresholds: HV = valence >= 7, LV = valence <= 3 (drop ambiguous middle).

Run:  ./venv/Scripts/python.exe train/case_valence_stratified.py
Cache: data/case_stratified.npz
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import train.results.valence.eevr_valence_stratified as evs  # noqa: E402
from train.results.valence.eevr_valence_stratified import (  # noqa: E402
    FAMILY_VASC_IDX,
    FAMILY_NONVASC_IDX,
    feats_family,
    feats_shared,
    loso,
)

# feats_family/feats_shared build timestamps and the band-pass from evs.FS (64 for
# EEVR). CASE BVP is downsampled to 100 Hz, so patch the module global to 100 BEFORE
# extraction so ts, the 0.5-4 Hz band, and the peak-distance are all correct.
evs.FS = 100

FS = 1000
DS = 10                       # downsample 1000 -> 100 Hz (matches case_extract.py)
FS_DS = FS // DS
WIN_S, STEP_S = 20, 10
HV_THR, LV_THR = 7.0, 3.0     # valence thresholds
HI_AROUSAL, LO_AROUSAL = 6.0, 4.0
PHYS = ROOT / "datasets" / "CASE_FULL" / "data" / "interpolated" / "physiological"
ANNO = ROOT / "datasets" / "CASE_FULL" / "data" / "interpolated" / "annotations"
CACHE = ROOT / "data" / "case_stratified.npz"


def extract_all() -> dict:
    """Window each subject's BVP; tag each window with mean valence + mean arousal."""
    Xa, Xb, val, aro, subj, vid = [], [], [], [], [], []
    files = sorted(PHYS.glob("sub_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    for path in files:
        sid = int(path.stem.split("_")[1])
        ph = pd.read_csv(path)
        an = pd.read_csv(ANNO / path.name)
        bvp = ph["bvp"].to_numpy(float)
        dt = ph["daqtime"].to_numpy(float)
        jt = an["jstime"].to_numpy(float)
        av = an["valence"].to_numpy(float)
        aa = an["arousal"].to_numpy(float)
        avid = an["video"].to_numpy(int)

        for v in np.unique(avid):
            amask = avid == v
            if amask.sum() < 5:
                continue
            t0, t1 = jt[amask][0], jt[amask][-1]
            pmask = (dt >= t0) & (dt <= t1)
            seg_bvp = bvp[pmask][::DS]
            seg_t = dt[pmask][::DS]
            w, st = WIN_S * FS_DS, STEP_S * FS_DS
            if len(seg_bvp) < w:
                continue
            for i in range(0, len(seg_bvp) - w + 1, st):
                seg = seg_bvp[i:i + w]
                wt0, wt1 = seg_t[i], seg_t[i + w - 1]
                wmask = amask & (jt >= wt0) & (jt <= wt1)
                if wmask.sum() < 3:
                    continue
                wval = float(np.mean(av[wmask]))
                waro = float(np.mean(aa[wmask]))
                fa = feats_family(seg)        # uses FS module-global -> patch below
                fb = feats_shared(seg)
                if fa is None or fb is None:
                    continue
                Xa.append(fa); Xb.append(fb)
                val.append(wval); aro.append(waro); subj.append(sid); vid.append(int(v))
        print(f"  sub_{sid}: {len(val)} windows so far")

    out = dict(
        Xa=np.asarray(Xa, float), Xb=np.asarray(Xb, float),
        valence=np.asarray(val, float), arousal=np.asarray(aro, float),
        subjects=np.asarray(subj, int), video=np.asarray(vid, int),
    )
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    print(f"  cached {len(val)} windows -> {CACHE.name}")
    return out


def load() -> dict:
    if CACHE.exists():
        print(f"Loading cached {CACHE.name}")
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return extract_all()


def _match_arousal(aro: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Subsample so HV and LV have overlapping arousal distributions (bin-match)."""
    bins = np.arange(0.5, 9.6, 1.0)
    keep = np.zeros(len(y), dtype=bool)
    for b0, b1 in zip(bins[:-1], bins[1:]):
        in_bin = (aro >= b0) & (aro < b1)
        pos = np.where(in_bin & (y == 1))[0]
        neg = np.where(in_bin & (y == 0))[0]
        n = min(len(pos), len(neg))
        if n == 0:
            continue
        keep[rng.choice(pos, n, replace=False)] = True
        keep[rng.choice(neg, n, replace=False)] = True
    return keep


def experiment(d, title, band_mask, do_match):
    """One HV-vs-LV contrast; reports arousal match + LOSO for both extractors."""
    val, aro, subj = d["valence"], d["arousal"], d["subjects"]
    hv = val >= HV_THR
    lv = val <= LV_THR
    sel = band_mask & (hv | lv)
    y = hv[sel].astype(int)
    a = aro[sel]
    g = subj[sel]
    Xa, Xb = d["Xa"][sel], d["Xb"][sel]
    vids = d["video"][sel]

    print(f"\n=== {title} ===")
    print(f"  windows: {len(y)} (HV={int(y.sum())}, LV={int((1-y).sum())}) "
          f"subjects={len(np.unique(g))}")
    if len(y) < 20 or y.sum() < 5 or (1 - y).sum() < 5:
        print("  (too few windows — skipped)")
        return
    # arousal of HV vs LV BEFORE matching (proof the band really equalizes arousal)
    a_hv, a_lv = a[y == 1], a[y == 0]
    t, p = ttest_ind(a_hv, a_lv, equal_var=False)
    print(f"  arousal HV={a_hv.mean():.2f} vs LV={a_lv.mean():.2f}  "
          f"(t={t:.2f}, p={p:.3f})  {'MATCHED' if p > 0.05 else 'DIFFERS -> matching'}")

    if do_match and p <= 0.05:
        keep = _match_arousal(a, y, np.random.default_rng(0))
        y, a, g, Xa, Xb, vids = y[keep], a[keep], g[keep], Xa[keep], Xb[keep], vids[keep]
        a_hv, a_lv = a[y == 1], a[y == 0]
        t, p = ttest_ind(a_hv, a_lv, equal_var=False)
        print(f"  after match: {len(y)} win, arousal HV={a_hv.mean():.2f} vs "
              f"LV={a_lv.mean():.2f} (p={p:.3f})")

    # independence: distinct videos per class
    nv_hv = len(np.unique(vids[y == 1])); nv_lv = len(np.unique(vids[y == 0]))
    print(f"  distinct videos: HV={nv_hv}, LV={nv_lv}")

    for tag, X in (("family", Xa), ("shared", Xb)):
        acc, nf = loso(X, y, g)
        print(f"  [{tag}] ALL          : {acc:5.1f}%  ({nf} folds)")
        if tag == "family":
            av, _ = loso(X[:, FAMILY_VASC_IDX], y, g)
            an, _ = loso(X[:, FAMILY_NONVASC_IDX], y, g)
            print(f"  [{tag}] vascular     : {av:5.1f}%")
            print(f"  [{tag}] fd+morph+hrv : {an:5.1f}%")
    return y, Xb, g


def permutation_test(X, y, g, n=500):
    """Is Exp1 above chance? Shuffle labels within nothing — full shuffle."""
    obs, _ = loso(X, y, g)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(n):
        yp = rng.permutation(y)
        if len(np.unique(yp)) < 2:
            continue
        s, _ = loso(X, yp, g)
        null.append(s)
    null = np.array(null)
    p = (np.sum(null >= obs) + 1) / (len(null) + 1)
    print(f"\n  permutation test Exp1: observed={obs:.1f}%, "
          f"null={null.mean():.1f}±{null.std():.1f}%, p={p:.3f}")


def main():
    d = load()
    aro = d["arousal"]
    print(f"\nwindows total: {len(aro)}, arousal range {aro.min():.1f}-{aro.max():.1f}")

    hi = experiment(d, "HV vs LV @ HIGH arousal (>=6) <- PRIMARY",
                    aro >= HI_AROUSAL, do_match=True)
    experiment(d, "HV vs LV @ LOW arousal (<=4)",
               aro <= LO_AROUSAL, do_match=True)
    experiment(d, "HV vs LV @ ALL arousal (plain valence baseline)",
               np.ones(len(aro), bool), do_match=False)

    if hi is not None:
        y, Xb, g = hi
        permutation_test(Xb, y, g)

    print("\nDone. Compare HIGH-band (arousal isolated) vs ALL (plain). "
          "If HIGH ~ chance but ALL ~60%, the ~60% was arousal, not valence.")


if __name__ == "__main__":
    main()
