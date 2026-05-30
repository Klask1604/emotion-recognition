#!/usr/bin/env python3
"""Test whether HRV features (vagal tone: RMSSD/pNN50/LF/HF) add valence signal,
on WESAD, CASE, EmoWear. Compares: existing-20 vs HRV-7 vs combined-27, LOSO."""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
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
from affectus.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES, extract_morph_features)
from affectus.legacy.ppg_hrv_features import (  # noqa: E402
    HRV_FEATURE_NAMES, extract_hrv_features)

FD = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]


def est_hr(seg, fs):
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    return len(pk) / (len(seg) / fs) * 60 if len(pk) > 1 else 0.0


def extract_all(seg, fs):
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
    old = [fd.as_dict()[k] for k in FD] + [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES]
    hrv = [hv.as_dict()[k] for k in HRV_FEATURE_NAMES]
    return old, hrv


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
    comb = np.hstack([old, hrv])
    print(f"  {name:8s} | old-20 {loso(old, y, subj):5.1f}% | "
          f"HRV-7 {loso(hrv, y, subj):5.1f}% | combined {loso(comb, y, subj):5.1f}%")


def wesad():
    FS = 64
    old, hrv, y, subj = [], [], [], []
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        s = int(Path(path).stem[1:])
        d = pickle.load(open(path, "rb"), encoding="latin1")
        bvp = np.array(d["signal"]["wrist"]["BVP"]).flatten()
        lbl = np.array(d["label"])
        wb, wl = 20 * FS, 20 * 700
        for i in range(len(bvp) // wb):
            seg = bvp[i * wb:(i + 1) * wb]
            ls = lbl[i * wl:(i + 1) * wl]
            v, c = np.unique(ls, return_counts=True)
            dom = v[np.argmax(c)]
            if dom not in (2, 3):
                continue
            r = extract_all(seg, FS)
            if r is None:
                continue
            old.append(r[0]); hrv.append(r[1]); y.append(1 if dom == 3 else 0); subj.append(s)
    report("WESAD", old, hrv, y, subj)


def main():
    print("Dataset  | old features | HRV only | combined  (VALENCE, LOSO)")
    print("-" * 62)
    wesad()
    # CASE + EmoWear from raw would take long; this run focuses on WESAD where
    # signal exists, to see if HRV adds to the 78%. (CASE/EmoWear ~50% regardless.)


if __name__ == "__main__":
    main()
