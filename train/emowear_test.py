#!/usr/bin/env python3
"""Run the valence-FD + morphological PPG features on the EmoWear dataset
(Empatica E4 BVP, 64 Hz) and classify valence with leave-one-subject-out.

EmoWear layout per subject folder:
  signals-e4-bvp.csv   timestamp,value  (64 Hz BVP)
  markers-phase2.csv   seq,exp,...,vidB,...,surveyB,...  (clip start/end)
  surveys.csv          seq,exp,valence,arousal,...       (SAM per clip)

For each clip we slice BVP between vidB (video begin) and surveyB (survey begin),
window it (20 s, 10 s step), extract features, and label valence (>=5 positive).
Works on the sample (1 subject -> feature sanity only) and the full set (LOSO).
"""

from __future__ import annotations

import csv
import glob
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
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from biofizic.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from biofizic.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)

FS = 64
WIN_S, STEP_S = 20, 10
FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]


def est_hr(seg: np.ndarray) -> float:
    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (FS / 2), 4 / (FS / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * FS))
    return len(pk) / (len(seg) / FS) * 60 if len(pk) > 1 else 0.0


def _read_csv(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def find_subject_dirs() -> list[Path]:
    # Either datasets/sample/csv/<subj>/ or datasets/EmoWear/csv/<subj>/
    for base in ["datasets/EmoWear/csv", "datasets/emowear/csv", "datasets/sample/csv"]:
        dirs = sorted((ROOT / base).glob("*/")) if (ROOT / base).exists() else []
        if dirs:
            return [d for d in dirs if (d / "signals-e4-bvp.csv").exists()]
    return []


def main() -> None:
    subj_dirs = find_subject_dirs()
    if not subj_dirs:
        print("No EmoWear subject folders found.")
        return
    print(f"Found {len(subj_dirs)} subject(s).")

    Xfd, Xmo, y, subj = [], [], [], []
    for si, sdir in enumerate(subj_dirs):
        bvp = np.loadtxt(sdir / "signals-e4-bvp.csv", delimiter=",", skiprows=1)
        t = bvp[:, 0]
        v = bvp[:, 1]
        markers = _read_csv(sdir / "markers-phase2.csv")
        surveys = {row["exp"]: row for row in _read_csv(sdir / "surveys.csv")}
        kept = 0
        for m in markers:
            exp = m.get("exp")
            if exp not in surveys:
                continue
            try:
                vid_b = float(m["vidB"])
                survey_b = float(m["surveyB"])
                valence = float(surveys[exp]["valence"])
            except (KeyError, ValueError):
                continue
            mask = (t >= vid_b) & (t < survey_b)
            seg_full = v[mask]
            if len(seg_full) < WIN_S * FS:
                continue
            label = 1 if valence >= 5.0 else 0
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
                Xfd.append([fd.as_dict()[k] for k in FD_KEYS])
                Xmo.append([mo.as_dict()[k] for k in MORPH_FEATURE_NAMES])
                y.append(label)
                subj.append(si)
                kept += 1
        print(f"  {sdir.name}: {kept} windows")

    Xfd, Xmo, y, subj = map(np.array, (Xfd, Xmo, y, subj))
    n_subj = len(np.unique(subj))
    print(f"\nTotal: n={len(y)}, positive={y.mean():.1%}, subjects={n_subj}")

    if n_subj < 2:
        print("Only 1 subject -> feature sanity OK, but no LOSO possible.")
        print("Download the full csv.zip for the cross-subject result.")
        return

    logo = LeaveOneGroupOut()

    def loso(X):
        b = []
        for tr, te in logo.split(X, y, groups=subj):
            if len(np.unique(y[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(),
                              SVC(kernel="rbf", class_weight="balanced"))
            c.fit(X[tr], y[tr])
            b.append(balanced_accuracy_score(y[te], c.predict(X[te])))
        return np.mean(b) * 100 if b else 0

    print(f"  FD only    : {loso(Xfd):.1f}%")
    print(f"  Morph only : {loso(Xmo):.1f}%")
    print(f"  FD + Morph : {loso(np.hstack([Xfd, Xmo])):.1f}%")


if __name__ == "__main__":
    main()
