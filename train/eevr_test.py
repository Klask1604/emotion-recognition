#!/usr/bin/env python3
"""Test valence on EEVR (VR-induced emotion, 37 subjects). The valence label is
encoded in the clip code: 'LV*' = low valence (negative), 'HV*' = high valence
(positive) — ground truth from the experiment design (clips pre-selected per
Russell quadrant), not noisy per-subject self-report. VR 360 induction is the
strongest of all datasets and matches the product's VR context.

Extracts FD + morph + HRV + vascular features, LOSO valence."""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import MORPH_FEATURE_NAMES, extract_morph_features  # noqa: E402
from affectus.legacy.ppg_hrv_features import HRV_FEATURE_NAMES, extract_hrv_features  # noqa: E402
from affectus.legacy.ppg_vascular_features import VASC_FEATURE_NAMES, extract_vascular_features  # noqa: E402

FD = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]
TARGET_FS = 100   # downsample target
WIN_S, STEP_S = 20, 10
OUT = ROOT / "data" / "eevr_features.npz"


def est_hr(seg, fs):
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    return len(pk) / (len(seg) / fs) * 60 if len(pk) > 1 else 0.0


def feats(seg, fs):
    ts = [int(j * 1000 / fs) for j in range(len(seg))]
    g = [int(round(x * 1000)) for x in seg]  # scale float PPG to int-ish
    hr = est_hr(seg, fs)
    if hr <= 0:
        return None
    fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
    mo = extract_morph_features(g, ts)
    hv = extract_hrv_features(g, ts)
    vc = extract_vascular_features(g, ts)
    if not (fd.valid and mo.valid and hv.valid and vc.valid):
        return None
    return ([fd.as_dict()[k] for k in FD] +
            [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES] +
            [hv.as_dict()[k] for k in HRV_FEATURE_NAMES] +
            [vc.as_dict()[k] for k in VASC_FEATURE_NAMES])


def main():
    print("Loading EEVR Raw_PPG (large)...")
    df = pd.read_csv(ROOT / "datasets/EEVR/Physiological_Data/Raw_PPG.csv",
                     usecols=["PPG", "Label", "Participant ID"])
    # valence from clip code: contains 'LV' (neg) or 'HV' (pos); skip Baseline
    df = df[~df["Label"].str.contains("Baseline", na=False)]
    df["valence"] = df["Label"].apply(
        lambda s: 1 if "HV" in str(s) else (0 if "LV" in str(s) else -1))
    df = df[df["valence"] >= 0]

    # Empatica E4 BVP = 64 Hz fixed (the Time column is aggregated, not a usable
    # timestamp). Keep 64 Hz (already in the watch range; no downsample needed).
    FS = 64
    X, y, subj = [], [], []
    for (pid, label), grp in df.groupby(["Participant ID", "Label"]):
        sig = grp["PPG"].to_numpy(float)
        if len(sig) < WIN_S * FS:
            continue
        val = int(grp["valence"].iloc[0])
        w, st = WIN_S * FS, STEP_S * FS
        for i in range(0, len(sig) - w + 1, st):
            r = feats(sig[i:i + w], FS)
            if r is None:
                continue
            X.append(r); y.append(val); subj.append(int(pid))
    X, y, subj = np.array(X), np.array(y), np.array(subj)
    np.savez(OUT, X=X, y=y, subjects=subj)
    print(f"n={len(y)}, subjects={len(np.unique(subj))}, positive={y.mean():.1%}")

    logo = LeaveOneGroupOut()
    names = (FD + MORPH_FEATURE_NAMES + HRV_FEATURE_NAMES + VASC_FEATURE_NAMES)
    n_fd_mo = len(FD) + len(MORPH_FEATURE_NAMES)
    n_hrv = len(HRV_FEATURE_NAMES)
    vasc_idx = list(range(n_fd_mo + n_hrv, len(names)))

    def loso(Xm):
        b = []
        for tr, te in logo.split(Xm, y, groups=subj):
            if len(np.unique(y[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
            c.fit(Xm[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], c.predict(Xm[te])))
        return np.mean(b) * 100 if b else 0

    print(f"  ALL features      : {loso(X):.1f}%")
    print(f"  vascular only     : {loso(X[:, vasc_idx]):.1f}%")
    print(f"  fd+morph+hrv      : {loso(X[:, :n_fd_mo + n_hrv]):.1f}%")


if __name__ == "__main__":
    main()
