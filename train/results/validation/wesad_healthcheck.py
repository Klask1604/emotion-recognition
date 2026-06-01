#!/usr/bin/env python3
"""Health check: run the valence-FD + morphological PPG features on WESAD wrist
BVP and try to classify stress vs baseline with leave-one-subject-out. If this
WORKS (>65%), the feature pipeline is sound and the DEAP-Kaggle failure was the
dataset; if it also flatlines (~50%), the method itself is structurally weak."""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)

FS_BVP = 64
FS_LBL = 700
WIN_S = 20
FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]


def est_hr(seg: np.ndarray) -> float:
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (FS_BVP / 2), 4 / (FS_BVP / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * FS_BVP))
    return len(pk) / (len(seg) / FS_BVP) * 60 if len(pk) > 1 else 0.0


def main() -> None:
    Xfd, Xmo, y, subj = [], [], [], []
    files = sorted(glob.glob(str(ROOT / "datasets" / "WESAD" / "S*" / "S*.pkl")))
    for path in files:
        sid = Path(path).stem  # "S10"
        s = int(sid[1:])
        d = pickle.load(open(path, "rb"), encoding="latin1")
        bvp = np.array(d["signal"]["wrist"]["BVP"]).flatten()
        lbl = np.array(d["label"])
        win_b, win_l = WIN_S * FS_BVP, WIN_S * FS_LBL
        for i in range(len(bvp) // win_b):
            seg = bvp[i * win_b:(i + 1) * win_b]
            lseg = lbl[i * win_l:(i + 1) * win_l]
            vals, cnts = np.unique(lseg, return_counts=True)
            dom = vals[np.argmax(cnts)]
            if dom not in (1, 2):  # baseline vs stress only
                continue
            hr = est_hr(seg)
            if hr <= 0:
                continue
            ts = [int(j * 1000 / FS_BVP) for j in range(len(seg))]
            g = [int(v) for v in seg]
            fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
            mo = extract_morph_features(g, ts)
            if not fd.valid or not mo.valid:
                continue
            fdd, mod = fd.as_dict(), mo.as_dict()
            Xfd.append([fdd[k] for k in FD_KEYS])
            Xmo.append([mod[k] for k in MORPH_FEATURE_NAMES])
            y.append(1 if dom == 2 else 0)
            subj.append(s)

    Xfd, Xmo, y, subj = map(np.array, (Xfd, Xmo, y, subj))
    print(f"WESAD stress-vs-baseline: n={len(y)}, stress={y.mean():.1%}, "
          f"subjects={len(np.unique(subj))}")

    logo = LeaveOneGroupOut()

    def loso(X):
        b = []
        for tr, te in logo.split(X, y, groups=subj):
            c = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", class_weight="balanced"))
            c.fit(X[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], c.predict(X[te])))
        return np.mean(b) * 100

    print(f"  FD only        : {loso(Xfd):.1f}%")
    print(f"  Morph only     : {loso(Xmo):.1f}%")
    print(f"  FD + Morph     : {loso(np.hstack([Xfd, Xmo])):.1f}%")
    print("\n  >65% => pipeline is sound (DEAP-Kaggle was the problem).")
    print("  ~50% => the method itself is structurally weak.")


if __name__ == "__main__":
    main()
