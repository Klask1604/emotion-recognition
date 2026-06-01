#!/usr/bin/env python3
"""Extract FD + morphological PPG features from the CASE dataset and cache to
data/case_features.npz. CASE has CONTINUOUS valence/arousal annotation (20 Hz
joystick), so each 20 s BVP window is labelled with the valence/arousal
ANNOTATED DURING THAT WINDOW — much cleaner than a single retrospective rating
per clip (the EmoWear/WPED weakness).

physiological CSV: daqtime(ms,1000Hz), ecg, bvp, gsr, ...
annotation  CSV: jstime(ms,20Hz), valence(0.5-9.5), arousal, video
"""

from __future__ import annotations

import glob
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)

FS = 1000           # BVP physiological sample rate
ANNO_FS = 20
WIN_S, STEP_S = 20, 10
FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]
PHYS = ROOT / "datasets" / "CASE_FULL" / "data" / "interpolated" / "physiological"
ANNO = ROOT / "datasets" / "CASE_FULL" / "data" / "interpolated" / "annotations"
OUT = ROOT / "data" / "case_features.npz"


def est_hr(seg: np.ndarray, fs: int) -> float:
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (fs / 2), 4 / (fs / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * fs))
    return len(pk) / (len(seg) / fs) * 60 if len(pk) > 1 else 0.0


def main() -> None:
    # Downsample BVP 1000 Hz -> 100 Hz to keep feature extraction fast and match
    # the watch range; harmonics (<5 Hz) are far below the 50 Hz Nyquist.
    DS = 10
    fs_ds = FS // DS

    X, val, aro, subj, trials = [], [], [], [], []
    tc = 0
    files = sorted(PHYS.glob("sub_*.csv"), key=lambda p: int(p.stem.split("_")[1]))
    for path in files:
        sid = int(path.stem.split("_")[1])
        ph = pd.read_csv(path)
        an = pd.read_csv(ANNO / path.name)
        bvp = ph["bvp"].to_numpy(float)
        dt = ph["daqtime"].to_numpy(float)          # ms, 1000 Hz
        jt = an["jstime"].to_numpy(float)           # ms, 20 Hz
        av = an["valence"].to_numpy(float)
        aa = an["arousal"].to_numpy(float)
        avid = an["video"].to_numpy(int)

        for vid in np.unique(avid):
            amask = avid == vid
            if amask.sum() < 5:
                continue
            t0, t1 = jt[amask][0], jt[amask][-1]      # this clip's time span (ms)
            pmask = (dt >= t0) & (dt <= t1)
            seg_bvp = bvp[pmask][::DS]                 # downsample to 100 Hz
            seg_t = dt[pmask][::DS]
            if len(seg_bvp) < WIN_S * fs_ds:
                continue
            w, st = WIN_S * fs_ds, STEP_S * fs_ds
            # Annotation lag: physiology reacts ~3 s before the joystick rating,
            # so the label for a PPG window at time t comes from the annotation
            # LAG_MS later (reviewer suggestion). Shift the label window forward.
            LAG_MS = 3000
            for i in range(0, len(seg_bvp) - w + 1, st):
                seg = seg_bvp[i:i + w]
                wt0, wt1 = seg_t[i], seg_t[i + w - 1]   # window time span
                # valence/arousal annotated during this window, shifted by lag
                wmask = amask & (jt >= wt0 + LAG_MS) & (jt <= wt1 + LAG_MS)
                if wmask.sum() < 3:
                    # fall back to unlagged if the lagged window runs off the clip
                    wmask = amask & (jt >= wt0) & (jt <= wt1)
                    if wmask.sum() < 3:
                        continue
                wval = float(np.mean(av[wmask]))
                waro = float(np.mean(aa[wmask]))
                hr = est_hr(seg, fs_ds)
                if hr <= 0:
                    continue
                ts = [int(j * 1000 / fs_ds) for j in range(len(seg))]
                g = [int(round(x)) for x in seg]
                fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
                mo = extract_morph_features(g, ts)
                if not fd.valid or not mo.valid:
                    continue
                base = ([fd.as_dict()[k] for k in FD_KEYS] +
                        [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES])
                # Delta/trend features (reviewer suggestion): the CHANGE of each
                # feature across the window — first half vs second half. Direction
                # of change can carry valence better than the absolute level.
                half = len(seg) // 2
                deltas = [0.0] * len(base)
                if half >= 64:
                    g1 = [int(round(x)) for x in seg[:half]]
                    g2 = [int(round(x)) for x in seg[half:]]
                    ts1 = [int(j * 1000 / fs_ds) for j in range(half)]
                    ts2 = [int(j * 1000 / fs_ds) for j in range(len(seg) - half)]
                    fd1 = extract_valence_fd_features(g1, ts1, hr_bpm=hr)
                    mo1 = extract_morph_features(g1, ts1)
                    fd2 = extract_valence_fd_features(g2, ts2, hr_bpm=hr)
                    mo2 = extract_morph_features(g2, ts2)
                    if fd1.valid and fd2.valid and mo1.valid and mo2.valid:
                        v1 = ([fd1.as_dict()[k] for k in FD_KEYS] +
                              [mo1.as_dict()[k] for k in MORPH_FEATURE_NAMES])
                        v2 = ([fd2.as_dict()[k] for k in FD_KEYS] +
                              [mo2.as_dict()[k] for k in MORPH_FEATURE_NAMES])
                        deltas = [b - a for a, b in zip(v1, v2)]
                X.append(base + deltas)
                val.append(wval)
                aro.append(waro)
                subj.append(sid)
                trials.append(tc)
            tc += 1
        print(f"  sub_{sid}: total so far {len(X)} windows")

    X = np.asarray(X, dtype=float)
    names = (FD_KEYS + MORPH_FEATURE_NAMES +
             [f"d_{k}" for k in FD_KEYS + MORPH_FEATURE_NAMES])
    out = OUT.parent / "case_features_v2.npz"  # lag + delta version
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=X, valence=np.array(val), arousal=np.array(aro),
             subjects=np.array(subj), trials=np.array(trials), feature_names=names)
    print(f"  (saved {out} with lag+delta)")
    print(f"\nSaved {OUT}: X={X.shape}, subjects={len(np.unique(subj))}")


if __name__ == "__main__":
    main()
