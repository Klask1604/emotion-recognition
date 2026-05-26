#!/usr/bin/env python3
"""
Train a WESAD stress RandomForest — for the thesis "ML on a foreign dataset"
comparison (a documented domain-shift negative result), NOT for production.

WESAD layout: datasets/WESAD/S<id>/S<id>.pkl, each a pickle with
  data["signal"]["chest"]["ECG"]  (RespiBAN, 700 Hz)
  data["label"]                    (700 Hz; 1=baseline 2=stress 3=amusement 4=meditation)
We derive IBI from chest-ECG R-peaks, slice 30 s windows inside single-label
regions, and reuse the production HRV extractor (compute_hrv_from_entries) so the
features match exactly what the live engine computes (train/serve parity).

Binary target: stress (label 2) vs non-stress (1, 3, 4). Leave-One-Subject-Out.

Usage:
    python train/train_wesad.py --data datasets/WESAD
"""

from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401
from biofizic.compute_features.hrv_metrics import compute_hrv_from_entries
from biofizic.ingestion.messages import InterbeatIntervalEntry
from biofizic.legacy.wesad import WESAD_FEATURE_NAMES, wesad_feature_vector

ECG_FS = 700
WINDOW_SEC = 30
STEP_SEC = 15
LABEL_BASELINE, LABEL_STRESS, LABEL_AMUSEMENT, LABEL_MEDITATION = 1, 2, 3, 4
USABLE_LABELS = (LABEL_BASELINE, LABEL_STRESS, LABEL_AMUSEMENT, LABEL_MEDITATION)


def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    """R-peak indices via band-pass (5-20 Hz) + squared signal + find_peaks."""
    from scipy.signal import butter, filtfilt, find_peaks

    nyq = fs / 2.0
    b, a = butter(2, [5.0 / nyq, 20.0 / nyq], btype="band")
    filtered = filtfilt(b, a, ecg)
    energy = filtered ** 2
    thr = energy.mean() + 0.5 * energy.std()
    peaks, _ = find_peaks(energy, distance=int(0.3 * fs), height=thr)
    return peaks


def ibi_entries_from_peaks(peaks: np.ndarray, fs: int) -> list[InterbeatIntervalEntry]:
    peak_ms = (peaks / fs * 1000.0).astype(int)
    entries = []
    for i in range(1, len(peak_ms)):
        entries.append(
            InterbeatIntervalEntry(interval_ms=int(peak_ms[i] - peak_ms[i - 1]), timestamp_ms=int(peak_ms[i]))
        )
    return entries


def windows_for_subject(ecg: np.ndarray, labels: np.ndarray, fs: int):
    """Yield (feature_vector, binary_label) for single-label 30 s windows."""
    win = WINDOW_SEC * fs
    step = STEP_SEC * fs
    for start in range(0, len(ecg) - win, step):
        seg_label = labels[start:start + win]
        # Require a pure-label window (one of the usable conditions).
        uniq = np.unique(seg_label)
        if uniq.size != 1 or int(uniq[0]) not in USABLE_LABELS:
            continue
        peaks = detect_r_peaks(ecg[start:start + win], fs)
        entries = ibi_entries_from_peaks(peaks, fs)
        metrics = compute_hrv_from_entries(entries)
        if metrics is None or not metrics.is_valid:
            continue
        binary = "stress" if int(uniq[0]) == LABEL_STRESS else "non_stress"
        yield wesad_feature_vector(metrics), binary


def load_wesad(data_dir: Path):
    X, y, groups = [], [], []
    subj_files = sorted(data_dir.glob("S*/S*.pkl"))
    if not subj_files:
        raise FileNotFoundError(f"No WESAD S*/S*.pkl under {data_dir}")
    for pkl in subj_files:
        with open(pkl, "rb") as fh:
            data = pickle.load(fh, encoding="latin1")
        ecg = np.asarray(data["signal"]["chest"]["ECG"], dtype=float).flatten()
        labels = np.asarray(data["label"]).flatten()
        sid = pkl.stem
        n = 0
        for feat, binary in windows_for_subject(ecg, labels, ECG_FS):
            X.append(feat); y.append(binary); groups.append(sid); n += 1
        print(f"{sid}: {n} windows")
    return np.array(X, float), np.array(y), np.array(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "datasets" / "WESAD"))
    ap.add_argument("--output", default=str(ROOT / "models" / "wesad_rf.joblib"))
    ap.add_argument("--report", default=str(ROOT / "eval_results" / "wesad_report.json"))
    args = ap.parse_args()

    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import LeaveOneGroupOut

    X, y, groups = load_wesad(Path(args.data))
    print(f"Total windows: {len(y)}, subjects: {len(set(groups))}, stress: {(y=='stress').sum()}")

    # Standardize features (the mean/std are saved and reused at inference).
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xn = (X - mean) / std

    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    for tr, te in logo.split(Xn, y, groups):
        clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
        clf.fit(Xn[tr], y[tr])
        y_true.extend(y[te].tolist())
        y_pred.extend(clf.predict(Xn[te]).tolist())
    acc = float(accuracy_score(y_true, y_pred))
    f1_stress = float(f1_score(y_true, y_pred, labels=["stress"], average="macro", zero_division=0))
    print(f"LOSO accuracy={acc:.3f}  stress F1={f1_stress:.3f}")
    print(classification_report(y_true, y_pred, zero_division=0))

    # Final model on all data (for the parallel engine).
    clf = RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42)
    clf.fit(Xn, y)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "classes": list(clf.classes_),
            "feature_names": WESAD_FEATURE_NAMES,
            "feature_mean": mean.tolist(),
            "feature_std": std.tolist(),
            "loso_accuracy": acc,
            "loso_f1_stress": f1_stress,
            "source": "WESAD_chest_ECG",
        },
        out,
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
