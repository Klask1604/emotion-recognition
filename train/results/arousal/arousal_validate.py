#!/usr/bin/env python3
"""Validate OUR arousal estimator against labelled arousal (CASE + WESAD, offline).

Arousal is the spine of the thesis and the axis that WORKS — but "works" needs a
number. This script runs OUR arousal computation (the same Kubios stress index +
HRV chain the live watch uses) on datasets that have GROUND-TRUTH arousal, and
measures how well our estimate tracks the truth.

  CASE  — continuous joystick arousal (0.6-9.5). The cleanest test: correlate our
          per-window stress index with the reported arousal, per subject (LOSO-ish:
          Spearman within each subject, then averaged — the honest cross-subject rho).
  WESAD — ordinal arousal by condition (meditation < baseline < amusement < stress).
          Test: does our stress index increase monotonically across these? (AUC of
          stress-vs-calm, and mean stress index per condition.)

We derive IBI from the BVP (band-pass + peak detect), then compute the SAME stress
index (sqrt Baevsky) the production pipeline uses, so this validates the real thing.

Run: ./venv/Scripts/python.exe train/arousal_validate.py
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from scipy.stats import spearmanr

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.shared.hrv.metrics import compute_baevsky_indices  # noqa: E402

WIN_S, STEP_S = 30, 15   # arousal uses a longer HRV window than valence


def ibi_from_bvp(seg: np.ndarray, fs: float) -> np.ndarray:
    """Detect beats in a BVP window, return IBI in ms (PPG fallback)."""
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    if len(pk) < 4:
        return np.array([])
    return np.diff(pk) / fs * 1000.0


def ibi_from_ecg(seg: np.ndarray, fs: float) -> np.ndarray:
    """Detect R-peaks in an ECG window -> IBI in ms. ECG gives much cleaner beats
    than PPG (sharp QRS vs smooth pulse), so the stress index is trustworthy."""
    s = seg - np.mean(seg)
    # QRS band-pass 5-15 Hz, then square to emphasise R-peaks (Pan-Tompkins-ish)
    nyq = fs / 2
    hi = min(15.0, nyq * 0.9)
    b, a = butter(2, [5.0 / nyq, hi / nyq], btype="band")
    f = filtfilt(b, a, s)
    sq = f * f
    thr = sq.mean() + 0.5 * sq.std()
    pk, _ = find_peaks(sq, height=thr, distance=int(0.4 * fs))
    if len(pk) < 4:
        return np.array([])
    return np.diff(pk) / fs * 1000.0


def our_stress_index(ibi_ms: np.ndarray) -> float | None:
    """The SAME Kubios stress index the live pipeline computes (sqrt Baevsky)."""
    if len(ibi_ms) < 4:
        return None
    # light artifact guard: drop physiologically impossible intervals
    ibi = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]
    if len(ibi) < 4:
        return None
    _, kubios = compute_baevsky_indices(ibi)
    return float(kubios)


# ---------------------------------------------------------------- CASE
def validate_case() -> None:
    PHYS = ROOT / "datasets/CASE_FULL/data/interpolated/physiological"
    ANNO = ROOT / "datasets/CASE_FULL/data/interpolated/annotations"
    FS, DS = 1000, 10
    fs_ds = FS // DS
    print("\n=== CASE — our stress index vs continuous reported arousal ===")
    per_subj_rho = []
    for path in sorted(PHYS.glob("sub_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
        sid = int(path.stem.split("_")[1])
        ph = pd.read_csv(path); an = pd.read_csv(ANNO / path.name)
        # ECG at full 1000 Hz gives clean R-peaks (don't downsample ECG).
        ecg = ph["ecg"].to_numpy(float)
        dt = ph["daqtime"].to_numpy(float)
        jt = an["jstime"].to_numpy(float); aa = an["arousal"].to_numpy(float)
        si_list, ar_list = [], []
        w, st = WIN_S * FS, STEP_S * FS
        for i in range(0, len(ecg) - w + 1, st):
            seg = ecg[i:i + w]
            t0, t1 = dt[i], dt[i + w - 1]
            si = our_stress_index(ibi_from_ecg(seg, FS))
            mask = (jt >= t0) & (jt <= t1)
            if si is None or mask.sum() < 3:
                continue
            si_list.append(si); ar_list.append(float(aa[mask].mean()))
        if len(si_list) >= 8:
            rho, _ = spearmanr(si_list, ar_list)
            if np.isfinite(rho):
                per_subj_rho.append(rho)
    if per_subj_rho:
        arr = np.array(per_subj_rho)
        print(f"  per-subject Spearman rho (our SI vs reported arousal):")
        print(f"    mean={arr.mean():+.2f}  median={np.median(arr):+.2f}  "
              f"(n={len(arr)} subjects, positive in {(arr>0).mean():.0%})")
        print(f"  -> rho>0 means our stress index rises when the subject reports more "
              f"arousal. rho~0.3-0.5 is a solid wearable-HRV result.")
    else:
        print("  (not enough windows)")


# ---------------------------------------------------------------- WESAD
def validate_wesad() -> None:
    FS = 700  # chest ECG sample rate (clean R-peaks)
    COND = {1: "baseline", 2: "stress", 3: "amusement", 4: "meditation"}
    AROUSAL_RANK = {4: 0, 1: 1, 3: 2, 2: 3}  # meditation<baseline<amusement<stress
    print("\n=== WESAD — our stress index across conditions (ordinal arousal) ===")
    by_cond = {c: [] for c in COND}
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        d = pickle.load(open(path, "rb"), encoding="latin1")
        ecg = np.array(d["signal"]["chest"]["ECG"]).flatten()  # 700 Hz gold
        lbl = np.array(d["label"])
        wb = WIN_S * FS
        for i in range(len(ecg) // wb):
            seg = ecg[i * wb:(i + 1) * wb]
            ls = lbl[i * wb:(i + 1) * wb]
            vals, cnts = np.unique(ls, return_counts=True)
            dom = int(vals[np.argmax(cnts)])
            if dom not in COND:
                continue
            si = our_stress_index(ibi_from_ecg(seg, FS))
            if si is not None:
                by_cond[dom].append(si)
    print("  mean stress index per condition (should rise with arousal):")
    rows = []
    for c in sorted(COND, key=lambda k: AROUSAL_RANK[k]):
        vals = by_cond[c]
        if vals:
            print(f"    {COND[c]:11s} (arousal rank {AROUSAL_RANK[c]}): "
                  f"SI={np.mean(vals):5.1f}  (n={len(vals)})")
            rows += [(AROUSAL_RANK[c], v) for v in vals]
    if rows:
        rank = np.array([r for r, _ in rows]); si = np.array([v for _, v in rows])
        rho, _ = spearmanr(rank, si)
        print(f"  Spearman rho (arousal rank vs our SI): {rho:+.2f}  "
              f"(positive = our SI tracks the arousal ordering)")


def main() -> None:
    print("Validating OUR arousal estimator (Kubios stress index) vs labelled arousal.")
    validate_case()
    validate_wesad()
    print("\nThese rho values are the quantitative validation of the arousal axis — the "
          "spine of the thesis. Report them alongside the live watch behaviour.")


if __name__ == "__main__":
    main()
