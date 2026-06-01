#!/usr/bin/env python3
"""Extract FD + morphological PPG features from EmoWear ONCE and cache to
data/emowear_features.npz, with both valence and arousal labels. After this,
any classifier experiment runs in seconds on the cached features instead of
re-processing the BVP (~minutes)."""

from __future__ import annotations

import csv
import glob
import sys
import warnings
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)

FS = 64
WIN_S, STEP_S = 20, 10
FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]
OUT = ROOT / "data" / "emowear_features.npz"


def est_hr(seg: np.ndarray) -> float:
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (FS / 2), 4 / (FS / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * FS))
    return len(pk) / (len(seg) / FS) * 60 if len(pk) > 1 else 0.0


def _read(p: Path) -> list[dict]:
    return list(csv.DictReader(open(p, newline="")))


def main() -> None:
    dirs = sorted(glob.glob(str(ROOT / "datasets" / "EmoWear" / "csv" / "*/")))
    X, val, aro, subj, trials = [], [], [], [], []
    trial_counter = 0
    for si, sd in enumerate(dirs):
        sd = Path(sd)
        try:
            bvp = np.loadtxt(sd / "signals-e4-bvp.csv", delimiter=",", skiprows=1)
            markers = _read(sd / "markers-phase2.csv")
            surveys = {r["exp"]: r for r in _read(sd / "surveys.csv")}
        except Exception:
            continue
        t, v = bvp[:, 0], bvp[:, 1]
        for m in markers:
            e = m.get("exp")
            if e not in surveys:
                continue
            try:
                vb, sb = float(m["vidB"]), float(m["surveyB"])
                valence = float(surveys[e]["valence"])
                arousal = float(surveys[e]["arousal"])
            except (KeyError, ValueError):
                continue
            seg_full = v[(t >= vb) & (t < sb)]
            if len(seg_full) < WIN_S * FS:
                continue
            w, st = WIN_S * FS, STEP_S * FS
            for i in range(0, len(seg_full) - w + 1, st):
                seg = seg_full[i:i + w]
                hr = est_hr(seg)
                if hr <= 0:
                    continue
                ts = [int(j * 1000 / FS) for j in range(len(seg))]
                g = [int(x) for x in seg]
                fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
                mo = extract_morph_features(g, ts)
                if not fd.valid or not mo.valid:
                    continue
                X.append([fd.as_dict()[k] for k in FD_KEYS] +
                         [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES])
                val.append(valence)
                aro.append(arousal)
                subj.append(si)
                trials.append(trial_counter)
            trial_counter += 1
        print(f"  {sd.name}: total so far {len(X)} windows")

    X = np.asarray(X, dtype=float)
    names = FD_KEYS + MORPH_FEATURE_NAMES
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, valence=np.array(val), arousal=np.array(aro),
             subjects=np.array(subj), trials=np.array(trials), feature_names=names)
    print(f"\nSaved {OUT}: X={X.shape}, subjects={len(np.unique(subj))}")


if __name__ == "__main__":
    main()
