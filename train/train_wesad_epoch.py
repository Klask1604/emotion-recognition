#!/usr/bin/env python3
"""
WESAD wrist evaluation — pipeline aliniat epoch_features.py (30s IBI).

Label binary: stress (TSST) vs non-stress (baseline + amusement).
Leave-One-Subject-Out; gate deploy în thresholds.py.

Usage:
    python train/train_wesad_epoch.py
    python train/train_wesad_epoch.py --eval-only
    python train/train_wesad_epoch.py --data ./wesad
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
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401
from biofizic.epoch_features import FEATURE_NAMES, compute_from_ibi_entries
from biofizic.thresholds import ML_GATE_ACCURACY_MIN, ML_GATE_F1_MIN

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("wesad_epoch")

LABEL_BASELINE = 1
LABEL_STRESS = 2
LABEL_AMUSEMENT = 3
LABEL_FS = 700
WINDOW_SEC = 30
STEP_SEC = 15
LABEL_PURITY = 0.75


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
    """Majority label in [t0, t1) ms mapped to binary stress."""
    # labels at 700 Hz — index from session start ms
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


def extract_subject(pkl_path: Path, ibi_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with open(pkl_path, "rb") as f:
        data = pickle.load(f, encoding="latin1")
    labels = data["label"].flatten()
    entries = _load_ibi_entries(ibi_path)
    if len(entries) < 4:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)

    ts_vals = [ts for _, ts in entries if ts is not None]
    if not ts_vals:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
    t_start = min(ts_vals)
    t_end = max(ts_vals)

    X_list: list[list[float]] = []
    y_list: list[int] = []
    step_ms = STEP_SEC * 1000
    win_ms = WINDOW_SEC * 1000
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
        if not (35 <= feats.mean_hr <= 160):
            t += step_ms
            continue
        X_list.append(feats.as_vector().tolist())
        y_list.append(y_bin)
        t += step_ms

    if not X_list:
        return np.empty((0, len(FEATURE_NAMES))), np.empty((0,), dtype=int)
    return np.array(X_list, dtype=float), np.array(y_list, dtype=int)


def normalize_subject(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, dict]:
    baseline_mask = y == 0
    if baseline_mask.sum() < 3:
        mean = X.mean(axis=0)
        std = X.std(axis=0) + 1e-8
    else:
        mean = X[baseline_mask].mean(axis=0)
        std = X[baseline_mask].std(axis=0) + 1e-8
    return (X - mean) / std, {"mean": mean.tolist(), "std": std.tolist()}


def expected_calibration_error(y_true: np.ndarray, proba: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (proba >= lo) & (proba < hi if i < n_bins - 1 else proba <= hi)
        if not np.any(mask):
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(proba[mask]))
        ece += abs(acc - conf) * np.sum(mask) / n
    return float(ece)


def run_loso(
    X: np.ndarray, y: np.ndarray, groups: np.ndarray
) -> dict:
    logo = LeaveOneGroupOut()
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    proba_all: list[float] = []
    per_subject: list[dict] = []

    for train_idx, test_idx in logo.split(X, y, groups):
        subj = str(groups[test_idx[0]])
        clf = RandomForestClassifier(
            n_estimators=120,
            max_depth=8,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X[train_idx], y[train_idx])
        pred = clf.predict(X[test_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(pred.tolist())
        proba_all.extend(proba.tolist())
        acc = accuracy_score(y[test_idx], pred)
        f1 = f1_score(y[test_idx], pred, pos_label=1, zero_division=0)
        per_subject.append({"subject": subj, "accuracy": acc, "f1_stress": f1, "n": len(test_idx)})

    y_true_arr = np.array(y_true_all)
    y_pred_arr = np.array(y_pred_all)
    proba_arr = np.array(proba_all)
    accuracy = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_stress = float(f1_score(y_true_arr, y_pred_arr, pos_label=1, zero_division=0))
    ece = expected_calibration_error(y_true_arr, proba_arr)
    report = classification_report(
        y_true_arr, y_pred_arr, target_names=["non_stress", "stress"], output_dict=True
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr).tolist()
    return {
        "accuracy": accuracy,
        "f1_stress": f1_stress,
        "ece": ece,
        "confusion_matrix": cm,
        "classification_report": report,
        "per_subject": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="WESAD epoch-aligned ML eval")
    parser.add_argument("--data", default=str(ROOT / "wesad"))
    parser.add_argument(
        "--output",
        default=str(ROOT / "models" / "model_v4.joblib"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "eval_results" / "wesad_loso_report.json"),
    )
    parser.add_argument("--eval-only", action="store_true")
    args = parser.parse_args()

    data_dir = Path(args.data)
    pkl_files = sorted(data_dir.glob("S*/S*.pkl"))
    if not pkl_files:
        log.error("Niciun .pkl în %s", data_dir)
        sys.exit(1)

    all_X: list[np.ndarray] = []
    all_y: list[np.ndarray] = []
    all_groups: list[np.ndarray] = []
    pop_rows: list[np.ndarray] = []

    for idx, pkl_path in enumerate(pkl_files):
        subj = pkl_path.stem
        ibi_path = pkl_path.parent / f"{subj}_E4_Data" / "IBI.csv"
        if not ibi_path.exists():
            log.warning("%s: IBI.csv lipsă", subj)
            continue
        X_raw, y = extract_subject(pkl_path, ibi_path)
        if len(X_raw) == 0:
            log.warning("%s: 0 ferestre", subj)
            continue
        X_norm, _stats = normalize_subject(X_raw, y)
        all_X.append(X_norm)
        all_y.append(y)
        all_groups.append(np.full(len(y), idx, dtype=int))
        pop_rows.append(X_raw[y == 0] if np.any(y == 0) else X_raw)
        log.info("%s: %d ferestre (stress=%d)", subj, len(y), int(y.sum()))

    if not all_X:
        log.error("Nicio fereastră validă")
        sys.exit(1)

    X = np.vstack(all_X)
    y = np.concatenate(all_y)
    groups = np.concatenate(all_groups)
    pop_mat = np.vstack(pop_rows)
    pop_mean = pop_mat.mean(axis=0)
    pop_std = pop_mat.std(axis=0) + 1e-8

    loso = run_loso(X, y, groups)
    gate_passed = (
        loso["accuracy"] >= ML_GATE_ACCURACY_MIN
        and loso["f1_stress"] >= ML_GATE_F1_MIN
    )

    report = {
        "pipeline": "epoch_features.compute_from_ibi_entries",
        "window_sec": WINDOW_SEC,
        "step_sec": STEP_SEC,
        "n_subjects": len(all_X),
        "n_windows": int(len(y)),
        "binary_label": "stress vs baseline+amusement",
        "gate": {
            "accuracy_min": ML_GATE_ACCURACY_MIN,
            "f1_stress_min": ML_GATE_F1_MIN,
            "accuracy": loso["accuracy"],
            "f1_stress": loso["f1_stress"],
            "ece": loso["ece"],
            "passed": gate_passed,
        },
        "loso": loso,
        "population_stats": {
            "feature_names": FEATURE_NAMES,
            "pop_mean": pop_mean.tolist(),
            "pop_std": pop_std.tolist(),
            "n_subjects": len(all_X),
            "source": "WESAD_wrist_IBI_epoch",
        },
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Raport: %s", report_path)

    md_path = report_path.with_suffix(".md")
    with open(md_path, "w") as f:
        f.write("# WESAD LOSO — epoch pipeline\n\n")
        f.write(f"- Accuracy: **{loso['accuracy']:.3f}** (gate min {ML_GATE_ACCURACY_MIN})\n")
        f.write(f"- F1 stress: **{loso['f1_stress']:.3f}** (gate min {ML_GATE_F1_MIN})\n")
        f.write(f"- ECE: {loso['ece']:.3f}\n")
        f.write(f"- Gate: **{'PASS' if gate_passed else 'FAIL'}**\n")
    log.info("Markdown: %s", md_path)

    if args.eval_only or not gate_passed:
        if not gate_passed:
            log.warning("Gate FAIL — ML rămâne OFF")
        sys.exit(0 if gate_passed else 2)

    clf = RandomForestClassifier(
        n_estimators=120,
        max_depth=8,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "feature_names": FEATURE_NAMES,
            "population_stats": report["population_stats"],
            "binary": True,
            "label_names": ["non_stress", "stress"],
            "source": "WESAD_epoch_pipeline",
        },
        out,
    )
    log.info("Model salvat: %s", out)


if __name__ == "__main__":
    main()
