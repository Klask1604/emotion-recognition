#!/usr/bin/env python3
"""
Extract PPG frequency-domain valence features from the DEAP dataset.

For every (subject, trial) we take the PPG channel (DEAP channel 38,
Plethysmograph, 128 Hz), slice it into the same 20 s windows used on the watch,
estimate HR per window (to anchor the harmonic bands), and run the SAME feature
extractor as the live system (affectus.research.valence.ppg_fd) — so the DEAP
features and the Galaxy-Watch features are identical by construction. The valence
label (DEAP scale 1-9) is binarised at 5 (the standard DEAP split: low vs high
valence).

Output: data/deap_valence_fd.npz with X (features), y (binary valence),
subjects (subject id per row) — ready for a leave-one-subject-out classifier.

Usage:
    ./venv/Scripts/python.exe train/deap_extract_features.py
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from affectus.research.valence.ppg_fd import (  # noqa: E402
    ValenceFdFeatures,
    extract_valence_fd_features,
)

DEAP_DIR = ROOT / "datasets" / "deap-dataset" / "data_preprocessed_python"
OUT = ROOT / "data" / "deap_valence_fd.npz"

PPG_CHANNEL = 38          # DEAP Plethysmograph
FS = 128                  # DEAP preprocessed sample rate
WINDOW_S = 20             # match the watch valence-FD epoch
STEP_S = 10               # 50% overlap -> more windows per trial
VALENCE_SPLIT = 5.0       # DEAP standard low/high valence threshold

FEATURE_ORDER = [
    "bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf",
]


def _estimate_hr(ppg: np.ndarray, fs: int) -> float:
    """HR (bpm) from PPG peak count, to anchor the harmonic bands."""
    sig = ppg - float(np.mean(ppg))
    nyq = fs / 2.0
    b, a = butter(2, [0.5 / nyq, 4.0 / nyq], btype="band")
    if len(sig) <= 3 * max(len(a), len(b)):
        return 0.0
    filt = filtfilt(b, a, sig)
    peaks, _ = find_peaks(filt, distance=int(0.4 * fs))
    dur_s = len(ppg) / fs
    if dur_s <= 0 or len(peaks) < 2:
        return 0.0
    return len(peaks) / dur_s * 60.0


def _features_vector(f: ValenceFdFeatures) -> list[float]:
    d = f.as_dict()
    return [d[k] for k in FEATURE_ORDER]


def main() -> None:
    files = sorted(DEAP_DIR.glob("s*.dat"))
    if not files:
        print(f"No DEAP files in {DEAP_DIR}")
        return

    X, y, subjects = [], [], []
    win = WINDOW_S * FS
    step = STEP_S * FS

    for path in files:
        subj = int(path.stem[1:])  # s01 -> 1
        with open(path, "rb") as fh:
            d = pickle.load(fh, encoding="latin1")
        data = d["data"]      # (40, 40, 8064)
        labels = d["labels"]  # (40, 4) -> valence is col 0
        n_trials = data.shape[0]
        kept = 0
        for t in range(n_trials):
            ppg_full = data[t, PPG_CHANNEL, :].astype(float)
            valence = float(labels[t, 0])
            label = 1 if valence >= VALENCE_SPLIT else 0
            # Slide windows over the trial.
            start = 0
            while start + win <= len(ppg_full):
                seg = ppg_full[start:start + win]
                start += step
                hr = _estimate_hr(seg, FS)
                if hr <= 0:
                    continue
                ts = [int(i * 1000 / FS) for i in range(len(seg))]
                green = [int(v) for v in seg]
                feats = extract_valence_fd_features(green, ts, hr_bpm=hr)
                if not feats.valid:
                    continue
                X.append(_features_vector(feats))
                y.append(label)
                subjects.append(subj)
                kept += 1
        print(f"  {path.stem}: {kept} windows")

    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int)
    subjects = np.asarray(subjects, dtype=int)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT, X=X, y=y, subjects=subjects, feature_names=FEATURE_ORDER)
    print(f"\nSaved {OUT}")
    print(f"  X={X.shape}  y={y.shape}  "
          f"pos(high valence)={y.mean():.2%}  subjects={len(np.unique(subjects))}")


if __name__ == "__main__":
    main()
