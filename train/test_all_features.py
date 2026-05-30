#!/usr/bin/env python3
"""Test every feature family — FD, morph, HRV, vascular — and their union, on
WESAD (valence) and CASE (valence + clear-label valence). Reports LOSO balanced
accuracy. Vascular = perfusion index / tone (deterministic mechanism)."""

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

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import MORPH_FEATURE_NAMES, extract_morph_features  # noqa: E402
from affectus.legacy.ppg_hrv_features import HRV_FEATURE_NAMES, extract_hrv_features  # noqa: E402
from affectus.legacy.ppg_vascular_features import VASC_FEATURE_NAMES, extract_vascular_features  # noqa: E402

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
    vc = extract_vascular_features(g, ts)
    if not (fd.valid and mo.valid and hv.valid and vc.valid):
        return None
    return {
        "fd": [fd.as_dict()[k] for k in FD],
        "morph": [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES],
        "hrv": [hv.as_dict()[k] for k in HRV_FEATURE_NAMES],
        "vasc": [vc.as_dict()[k] for k in VASC_FEATURE_NAMES],
    }


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


def report(name, F, y, subj):
    y, subj = np.array(y), np.array(subj)
    def cat(*keys):
        return np.hstack([np.array([f[k] for f in F]) for k in keys])
    print(f"\n  {name} (n={len(y)}):")
    print(f"    vascular only        : {loso(cat('vasc'), y, subj):5.1f}%")
    print(f"    old (fd+morph)       : {loso(cat('fd','morph'), y, subj):5.1f}%")
    print(f"    + HRV                : {loso(cat('fd','morph','hrv'), y, subj):5.1f}%")
    print(f"    + HRV + vascular ALL : {loso(cat('fd','morph','hrv','vasc'), y, subj):5.1f}%")


def wesad():
    FS = 64
    F, y, subj = [], [], []
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
            r = feats(seg, FS)
            if r is None:
                continue
            F.append(r); y.append(1 if dom == 3 else 0); subj.append(s)
    report("WESAD valence", F, y, subj)


def case():
    FS, DS = 100, 10
    PHYS = ROOT / "datasets/CASE_FULL/data/interpolated/physiological"
    ANNO = ROOT / "datasets/CASE_FULL/data/interpolated/annotations"
    F, val, subj = [], [], []
    for path in sorted(PHYS.glob("sub_*.csv"), key=lambda p: int(p.stem.split("_")[1])):
        sid = int(path.stem.split("_")[1])
        ph = pd.read_csv(path); an = pd.read_csv(ANNO / path.name)
        bvp = ph["bvp"].to_numpy(float); dt = ph["daqtime"].to_numpy(float)
        jt = an["jstime"].to_numpy(float); av = an["valence"].to_numpy(float); avid = an["video"].to_numpy(int)
        for vid in np.unique(avid):
            am = avid == vid
            if am.sum() < 5:
                continue
            t0, t1 = jt[am][0], jt[am][-1]
            pm = (dt >= t0) & (dt <= t1)
            sb = bvp[pm][::DS]; st = dt[pm][::DS]
            w = 20 * FS
            for i in range(0, len(sb) - w + 1, w // 2):
                seg = sb[i:i + w]; wt0, wt1 = st[i], st[i + w - 1]
                wm = am & (jt >= wt0) & (jt <= wt1)
                if wm.sum() < 3:
                    continue
                r = feats(seg, FS)
                if r is None:
                    continue
                F.append(r); val.append(float(np.mean(av[wm]))); subj.append(sid)
    val = np.array(val)
    report("CASE valence (all)", F, (val >= 5).astype(int), subj)
    # clear labels
    mask = (val <= 3) | (val >= 7)
    Fc = [f for f, m in zip(F, mask) if m]
    report("CASE valence (clear)", Fc, (val[mask] >= 5).astype(int), np.array(subj)[mask])


def main():
    print("=== All feature families, LOSO valence ===")
    wesad()
    case()


if __name__ == "__main__":
    main()
