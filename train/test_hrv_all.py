#!/usr/bin/env python3
"""Complete the HRV-features table: WESAD, EmoWear, WPED, DEAP — old vs HRV vs
combined, valence LOSO. Confirms whether HRV adds signal only where it exists."""

from __future__ import annotations

import glob
import pickle
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

from biofizic.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from biofizic.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES, extract_morph_features)
from biofizic.legacy.ppg_hrv_features import (  # noqa: E402
    HRV_FEATURE_NAMES, extract_hrv_features)

FD = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]


def est_hr(seg, fs):
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    return len(pk) / (len(seg) / fs) * 60 if len(pk) > 1 else 0.0


def feats(seg, fs):
    ts = [int(j * 1000 / fs) for j in range(len(seg))]
    g = [int(round(x)) for x in seg]
    hr = est_hr(seg, fs)
    if hr <= 0:
        return None
    fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
    mo = extract_morph_features(g, ts)
    hv = extract_hrv_features(g, ts)
    if not (fd.valid and mo.valid and hv.valid):
        return None
    return ([fd.as_dict()[k] for k in FD] + [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES],
            [hv.as_dict()[k] for k in HRV_FEATURE_NAMES])


def loso(X, y, g):
    logo = LeaveOneGroupOut()
    b = []
    for tr, te in logo.split(X, y, groups=g):
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        b.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return np.mean(b) * 100 if b else 0


def report(name, old, hrv, y, subj):
    old, hrv, y, subj = map(np.array, (old, hrv, y, subj))
    print(f"  {name:8s} (n={len(y)}) | old {loso(old, y, subj):5.1f}% | "
          f"HRV {loso(hrv, y, subj):5.1f}% | combined {loso(np.hstack([old, hrv]), y, subj):5.1f}%")


def emowear():
    FS = 64
    old, hrv, y, subj = [], [], [], []
    import csv
    dirs = sorted(glob.glob(str(ROOT / "datasets/EmoWear/csv/*/")))
    for si, sd in enumerate(dirs):
        sd = Path(sd)
        try:
            bvp = np.loadtxt(sd / "signals-e4-bvp.csv", delimiter=",", skiprows=1)
            mk = list(csv.DictReader(open(sd / "markers-phase2.csv", newline="")))
            sv = {r["exp"]: r for r in csv.DictReader(open(sd / "surveys.csv", newline=""))}
        except Exception:
            continue
        t, v = bvp[:, 0], bvp[:, 1]
        for m in mk:
            e = m.get("exp")
            if e not in sv:
                continue
            try:
                vb, sb = float(m["vidB"]), float(m["surveyB"]); val = float(sv[e]["valence"])
            except (KeyError, ValueError):
                continue
            seg_full = v[(t >= vb) & (t < sb)]
            if len(seg_full) < 20 * FS:
                continue
            for i in range(0, len(seg_full) - 20 * FS + 1, 10 * FS):
                r = feats(seg_full[i:i + 20 * FS], FS)
                if r is None:
                    continue
                old.append(r[0]); hrv.append(r[1]); y.append(1 if val >= 5 else 0); subj.append(si)
    report("EmoWear", old, hrv, y, subj)


def wped():
    FS = 176
    VAL = {"joy": 1, "relaxed": 1, "anger": 0, "sadness": 0}
    old, hrv, y, subj = [], [], [], []
    for path in sorted(glob.glob(str(ROOT / "datasets/ppg_dataset_custom/Subject_*/*/*.txt"))):
        emo = Path(path).stem.split("_")[0]
        if emo not in VAL:
            continue
        s = int(Path(path).parent.parent.name.split("_")[1])
        sig = np.loadtxt(path)
        for i in range(0, len(sig) - 20 * FS + 1, 10 * FS):
            r = feats(sig[i:i + 20 * FS], FS)
            if r is None:
                continue
            old.append(r[0]); hrv.append(r[1]); y.append(VAL[emo]); subj.append(s)
    report("WPED", old, hrv, y, subj)


def main():
    print("Dataset            | old features | HRV only | combined  (VALENCE, LOSO)")
    print("-" * 70)
    emowear()
    wped()


if __name__ == "__main__":
    main()
