#!/usr/bin/env python3
"""
WESAD emotion ML cu features delta (z-score față de baseline sesiune).

Usage:
    python train/train_wesad_delta.py --data ./wesad
    python train/train_wesad_delta.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401
from biofizic.epoch_features import compute_from_ibi_entries
from biofizic.thresholds import ML_GATE_ACCURACY_MIN, ML_GATE_F1_MIN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("wesad_delta")

LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3
LABEL_FS = 700
WINDOW_SEC = 30
STEP_SEC = 15
LABEL_PURITY = 0.75

DELTA_FEATURE_NAMES = ["z_rmssd", "z_si", "z_hr", "z_mean_ibi"]


def _load_ibi_entries(ibi_path: Path) -> list[tuple[int, int | None]]:
    lines = ibi_path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    if len(lines) < 2:
        return []
    start_ms = int(float(lines[0].split(",")[0].strip()) * 1000)
    entries: list[tuple[int, int | None]] = []
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) < 2:
            continue
        try:
            offset_sec = float(parts[0].strip())
            ibi_sec = float(parts[1].strip())
        except ValueError:
            continue
        if ibi_sec <= 0:
            continue
        ibi_ms = int(round(ibi_sec * 1000))
        if 300 <= ibi_ms <= 2000:
            ts_ms = start_ms + int(round(offset_sec * 1000))
            entries.append((ibi_ms, ts_ms))
    return entries


def _window_label(labels: np.ndarray, t0: int, t1: int) -> tuple[int | None, float]:
    i0 = max(0, int(t0 * LABEL_FS / 1000))
    i1 = min(len(labels), int(t1 * LABEL_FS / 1000))
    if i1 <= i0:
        return None, 0.0
    win = labels[i0:i1]
    unique, counts = np.unique(win, return_counts=True)
    majority = int(unique[np.argmax(counts)])
    purity = float(counts.max() / len(win))
    if majority == LABEL_STRESS:
        return 1, purity
    if majority in (LABEL_BASELINE, LABEL_AMUSEMENT):
        return 0, purity
    return None, purity


def extract_subject(pkl_path: Path, ibi_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    labels = data["label"].flatten()
    entries = _load_ibi_entries(ibi_path)
    if len(entries) < 4:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, 4))

    ts_vals = [ts for _, ts in entries if ts is not None]
    t_start, t_end = min(ts_vals), max(ts_vals)
    step_ms = STEP_SEC * 1000
    win_ms = WINDOW_SEC * 1000

    raw_rows: list[list[float]] = []
    y_list: list[int] = []
    t = t_start
    while t + win_ms <= t_end:
        t1 = t + win_ms
        win_entries = [(ms, ts) for ms, ts in entries if ts is not None and t <= ts < t1]
        y_bin, purity = _window_label(labels, t - t_start, t1 - t_start)
        if y_bin is None or purity < LABEL_PURITY or len(win_entries) < 4:
            t += step_ms
            continue
        feats = compute_from_ibi_entries(win_entries)
        if feats is None or not feats.hrv_ready:
            t += step_ms
            continue
        raw_rows.append([feats.rmssd, feats.stress_index, feats.mean_hr, feats.mean_ibi_ms])
        y_list.append(y_bin)
        t += step_ms

    if not raw_rows:
        return np.empty((0, 4)), np.empty((0,)), np.empty((0, 4))

    raw = np.array(raw_rows, dtype=float)
    y = np.array(y_list, dtype=int)
    baseline_mask = y == 0
    if baseline_mask.sum() < 3:
        mu = raw.mean(axis=0)
        sd = raw.std(axis=0) + 1e-8
    else:
        mu = raw[baseline_mask].mean(axis=0)
        sd = raw[baseline_mask].std(axis=0) + 1e-8
    delta = (raw - mu) / sd
    return delta, y, raw


def run_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    logo = LeaveOneGroupOut()
    y_true, y_pred = [], []
    per_subject = []
    for tr, te in logo.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true.extend(y[te].tolist())
        y_pred.extend(pred.tolist())
        acc = accuracy_score(y[te], pred)
        f1 = f1_score(y[te], pred, pos_label=1, zero_division=0)
        per_subject.append({"subject": str(groups[te[0]]), "accuracy": acc, "f1_stress": f1})
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    return {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "f1_stress": float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0)),
        "classification_report": classification_report(
            y_true_arr, y_pred_arr, target_names=["non_stress", "stress"], output_dict=True
        ),
        "per_subject": per_subject,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "wesad"))
    p.add_argument("--output", default=str(ROOT / "models" / "emotion_delta_wesad.joblib"))
    p.add_argument("--report", default=str(ROOT / "eval_results" / "wesad_delta_report.json"))
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data)
    pkl_files = sorted(data_dir.glob("S*/S*.pkl"))
    all_X, all_y, all_g = [], [], []
    for idx, pkl_path in enumerate(pkl_files):
        ibi_path = pkl_path.parent / f"{pkl_path.stem}_E4_Data" / "IBI.csv"
        if not ibi_path.exists():
            continue
        X, y, _ = extract_subject(pkl_path, ibi_path)
        if len(X) == 0:
            continue
        all_X.append(X)
        all_y.append(y)
        all_g.append(np.full(len(y), idx))
        log.info("%s: %d windows", pkl_path.stem, len(y))

    if not all_X:
        log.error("No WESAD windows")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_g)
    loso = run_loso(X, y, groups)
    gate_passed = (
        loso["accuracy"] >= ML_GATE_ACCURACY_MIN
        and loso["f1_stress"] >= ML_GATE_F1_MIN
    )

    report = {
        "pipeline": "delta_features_per_subject_baseline",
        "feature_names": DELTA_FEATURE_NAMES,
        "gate": {
            "accuracy_min": ML_GATE_ACCURACY_MIN,
            "f1_stress_min": ML_GATE_F1_MIN,
            **loso,
            "passed": gate_passed,
        },
        "loso": loso,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report %s gate=%s", report_path, gate_passed)

    md = report_path.with_suffix(".md")
    with open(md, "w") as f:
        f.write(f"# WESAD Delta LOSO\n\n- Accuracy: {loso['accuracy']:.3f}\n")
        f.write(f"- F1 stress: {loso['f1_stress']:.3f}\n- Gate: {'PASS' if gate_passed else 'FAIL'}\n")

    if args.eval_only or not gate_passed:
        sys.exit(0 if gate_passed else 2)

    clf = RandomForestClassifier(
        n_estimators=120, max_depth=8, class_weight="balanced", random_state=42
    )
    clf.fit(X, y)
    joblib.dump({"model": clf, "feature_names": DELTA_FEATURE_NAMES}, args.output)
    log.info("Model %s", args.output)


if __name__ == "__main__":
    main()
