#!/usr/bin/env python3
"""EEVR valence with EDA added to PPG — multimodal HVHA-vs-LVHA.

Exp 1 (PPG only) reached only 56% on the arousal-matched valence contrast
(HVHA-vs-LVHA). This script tests whether adding electrodermal activity (EDA,
the skin-conductance channel EEVR also recorded) recovers any valence signal.

EDA is primarily AROUSAL-driven (sympathetic sweat response), so on an
arousal-MATCHED contrast it is expected to add little — but EEVR ships it and
it is the only remaining channel that might help, so we measure it honestly.

PPG and EDA are sampled at different rates (BVP 64 Hz, EDA 4 Hz) but share the
same clip label per (subject, Label). We window BOTH on the same 20 s / 10 s
grid per clip so window i of PPG aligns in time with window i of EDA, then
concatenate. EDA features per window:
  scl_mean, scl_std, scl_slope   — tonic level (SCL), linear trend
  scr_count, scr_amp_mean        — phasic responses (SCR peaks) after detrending
  eda_range                      — max-min span

FS hard-coded (E4: BVP 64 Hz, EDA 4 Hz); the Time column is unusable (false rate).

Run:  ./venv/Scripts/python.exe train/eevr_eda_multimodal.py
Cache: data/eevr_multimodal.npz
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from train.eevr_valence_stratified import (  # noqa: E402
    QUADRANTS,
    STEP_S,
    WIN_S,
    _quadrant,
    feats_family,
    feats_shared,
    comparison as _ppg_comparison,
    loso,
)

FS_PPG = 64
FS_EDA = 4
PPG_CSV = ROOT / "datasets/EEVR/Physiological_Data/Raw_PPG.csv"
EDA_CSV = ROOT / "datasets/EEVR/Physiological_Data/Raw_EDA.csv"
CACHE = ROOT / "data" / "eevr_multimodal.npz"
EDA_NAMES = ["scl_mean", "scl_std", "scl_slope", "scr_count", "scr_amp_mean", "eda_range"]


def eda_feats(seg: np.ndarray) -> list[float] | None:
    """Tonic (SCL) + phasic (SCR) features from a 20 s EDA window at 4 Hz."""
    if len(seg) < WIN_S * FS_EDA // 2:
        return None
    t = np.arange(len(seg)) / FS_EDA
    scl_mean = float(seg.mean())
    scl_std = float(seg.std())
    # linear trend (tonic slope) via least squares
    slope = float(np.polyfit(t, seg, 1)[0]) if len(seg) > 2 else 0.0
    # phasic: detrend, count positive peaks (SCRs) above a small amplitude
    detr = seg - np.polyval(np.polyfit(t, seg, 1), t)
    pk, props = find_peaks(detr, height=0.01, distance=int(1.0 * FS_EDA))
    scr_count = float(len(pk))
    scr_amp = float(props["peak_heights"].mean()) if len(pk) else 0.0
    eda_range = float(seg.max() - seg.min())
    out = [scl_mean, scl_std, slope, scr_count, scr_amp, eda_range]
    return out if all(np.isfinite(out)) else None


def extract_all() -> dict:
    print(f"Loading EEVR PPG ({PPG_CSV.stat().st_size / 1e6:.0f} MB) + EDA "
          f"({EDA_CSV.stat().st_size / 1e6:.0f} MB)...")
    ppg = pd.read_csv(PPG_CSV, usecols=["PPG", "Label", "Participant ID"])
    eda = pd.read_csv(EDA_CSV, usecols=["EDA", "Label", "Participant ID"])
    ppg["quad"] = ppg["Label"].map(_quadrant)
    eda["quad"] = eda["Label"].map(_quadrant)
    ppg = ppg[ppg["quad"].notna()]
    eda = eda[eda["quad"].notna()]

    # index EDA by (subject, label) for aligned per-clip slicing
    eda_groups = {k: g["EDA"].to_numpy(float)
                  for k, g in eda.groupby(["Participant ID", "Label"])}

    Xa, Xb, Xe, quad, subj = [], [], [], [], []
    wp, sp = WIN_S * FS_PPG, STEP_S * FS_PPG     # PPG window/step (samples)
    we, se = WIN_S * FS_EDA, STEP_S * FS_EDA     # EDA window/step (samples)
    for (pid, label), grp in ppg.groupby(["Participant ID", "Label"]):
        sig = grp["PPG"].to_numpy(float)
        eda_sig = eda_groups.get((pid, label))
        if eda_sig is None or len(sig) < wp:
            continue
        q = grp["quad"].iloc[0]
        n_win = (len(sig) - wp) // sp + 1
        for k in range(n_win):
            seg = sig[k * sp:k * sp + wp]
            # time-aligned EDA window (same clip-relative seconds)
            e0, e1 = k * se, k * se + we
            if e1 > len(eda_sig):
                continue
            eda_seg = eda_sig[e0:e1]
            fa = feats_family(seg)
            fb = feats_shared(seg)
            fe = eda_feats(eda_seg)
            if fa is None or fb is None or fe is None:
                continue
            Xa.append(fa); Xb.append(fb); Xe.append(fe)
            quad.append(q); subj.append(int(pid))

    out = dict(
        Xa=np.asarray(Xa, float), Xb=np.asarray(Xb, float),
        Xe=np.asarray(Xe, float), quad=np.asarray(quad),
        subjects=np.asarray(subj, int),
    )
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    print(f"  cached {len(quad)} windows (PPG+EDA aligned) -> {CACHE.name}")
    return out


def load() -> dict:
    if CACHE.exists():
        print(f"Loading cached {CACHE.name}")
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return extract_all()


def main():
    d = load()
    quad, subj = d["quad"], d["subjects"]
    print("window counts per quadrant:",
          {q: int((quad == q).sum()) for q in QUADRANTS})

    # HVHA vs LVHA only (the arousal-matched valence contrast that matters).
    mask = np.isin(quad, ("HVHA", "LVHA"))
    y = (quad[mask] == "HVHA").astype(int)
    g = subj[mask]
    print(f"\n=== HVHA vs LVHA — PPG vs PPG+EDA ===")
    print(f"  windows: {len(y)}  (pos={int(y.sum())}, neg={int((1-y).sum())})  "
          f"subjects={len(np.unique(g))}")

    Xppg_shared = d["Xb"][mask]
    Xeda = d["Xe"][mask]
    Xboth = np.hstack([Xppg_shared, Xeda])

    acc_ppg, nf = loso(Xppg_shared, y, g)
    acc_eda, _ = loso(Xeda, y, g)
    acc_both, _ = loso(Xboth, y, g)
    print(f"  PPG only (shared)   : {acc_ppg:5.1f}%   ({nf} folds)")
    print(f"  EDA only            : {acc_eda:5.1f}%")
    print(f"  PPG + EDA           : {acc_both:5.1f}%")
    print(f"\n  delta from adding EDA: {acc_both - acc_ppg:+.1f} pp")
    print("\nDone. Target was 65%.")


if __name__ == "__main__":
    main()
