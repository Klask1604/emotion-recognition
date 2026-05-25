#!/usr/bin/env python3
"""
Antrenare HAR pe WISDM smartwatch (acc + gyro) — aliniat Solid/GW7.

Usage:
    python train/train_wisdm_har.py --data ./datasets/wisdm
    python train/train_wisdm_har.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import zipfile
from pathlib import Path

import joblib
import numpy as np
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401
from biofizic.motion.motion_features import MOTION_FEATURE_NAMES, MotionFeatureVector

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
log = logging.getLogger("wisdm_har")

WISDM_URL = (
    "https://archive.ics.uci.edu/static/public/507/"
    "wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset.zip"
)

WINDOW_SEC = 30
STEP_SEC = 15
SAMPLE_HZ = 20
GATE_ACCURACY = 0.85
GATE_F1_WALK = 0.80

# WISDM activity letter -> Solid class
ACTIVITY_MAP = {
    "A": "WALK",  # walking
    "B": "WALK",  # jogging
    "C": "WALK",  # stairs
    "D": "STILL",  # sitting
    "E": "STILL",  # standing
    "F": "SCROLL",  # typing
    "P": "SCROLL",  # writing
    "G": "HAND",
    "H": "HAND",
    "I": "HAND",
    "J": "HAND",
    "K": "HAND",
    "L": "HAND",
    "M": "WALK",
    "N": "HAND",
    "O": "HAND",
    "Q": "HAND",
    "R": "HAND",
    "S": "HAND",
    "T": "HAND",
    "U": "HAND",
    "V": "STILL",
    "W": "HAND",
    "X": "WALK",
    "Y": "WALK",
    "Z": "HAND",
}


def _map_label(code: str) -> str | None:
    c = code.strip().upper()
    if len(c) != 1:
        return None
    return ACTIVITY_MAP.get(c)


def _parse_sensor_file(path: Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return (timestamps_ms, magnitudes, activity_codes)."""
    ts_list: list[int] = []
    mag_list: list[float] = []
    act_list: list[str] = []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip().rstrip(";")
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        try:
            act = parts[1]
            ts = int(parts[2])
            x, y, z = float(parts[-3]), float(parts[-2]), float(parts[-1])
        except ValueError:
            continue
        mapped = _map_label(act)
        if mapped is None:
            continue
        mag = float(np.sqrt(x * x + y * y + z * z))
        ts_list.append(ts)
        mag_list.append(mag)
        act_list.append(mapped)
    if not ts_list:
        return np.array([]), np.array([]), []
    order = np.argsort(ts_list)
    return (
        np.array(ts_list, dtype=np.int64)[order],
        np.array(mag_list, dtype=float)[order],
        [act_list[i] for i in order],
    )


def _find_watch_files(data_dir: Path, sensor: str) -> list[Path]:
    """sensor: 'accel' or 'gyro' (WISDM raw/watch layout)."""
    aliases = {
        "accelerometer": "accel",
        "accel": "accel",
        "gyroscope": "gyro",
        "gyro": "gyro",
    }
    key = aliases.get(sensor, sensor)
    patterns = [
        f"**/raw/watch/{key}/*.txt",
        f"**/watch/{key}/*.txt",
        f"**/*_{key}_watch.txt",
        f"**/watch/{sensor}/**/*.txt",
        f"**/*watch*{sensor}*/*.txt",
    ]
    found: list[Path] = []
    for pat in patterns:
        found.extend(data_dir.glob(pat))
    return sorted(set(p for p in found if p.is_file() and not p.name.startswith(".")))


def _subject_id(path: Path) -> str:
    m = re.search(r"(\d{4,6})", path.stem)
    return m.group(1) if m else path.stem


def _windows_for_subject(
    acc_path: Path, gyro_path: Path | None
) -> tuple[list[np.ndarray], list[str], str]:
    ts_a, mag_a, act_a = _parse_sensor_file(acc_path)
    if len(ts_a) < SAMPLE_HZ * WINDOW_SEC:
        return [], [], _subject_id(acc_path)
    if gyro_path and gyro_path.exists():
        ts_g, mag_g, _ = _parse_sensor_file(gyro_path)
    else:
        ts_g, mag_g = ts_a, np.zeros_like(mag_a)

    t0, t1 = int(ts_a[0]), int(ts_a[-1])
    step_ns = int(STEP_SEC * 1e9)
    win_ns = int(WINDOW_SEC * 1e9)
    X, y = [], []
    t = t0
    while t + win_ns <= t1:
        mask_a = (ts_a >= t) & (ts_a < t + win_ns)
        if mask_a.sum() < SAMPLE_HZ * 10:
            t += step_ns
            continue
        acts = [act_a[i] for i in np.where(mask_a)[0]]
        if not acts:
            t += step_ns
            continue
        label = max(set(acts), key=acts.count)
        seg_a = mag_a[mask_a]
        if gyro_path and len(ts_g):
            mask_g = (ts_g >= t) & (ts_g < t + win_ns)
            seg_g = mag_g[mask_g] if mask_g.sum() else np.zeros(len(seg_a))
        else:
            seg_g = np.zeros(len(seg_a))
        feat = MotionFeatureVector.from_wisdm_window(seg_a, seg_g, sample_hz=SAMPLE_HZ)
        if feat is not None:
            X.append(feat.values)
            y.append(label)
        t += step_ns
    return X, y, _subject_id(acc_path)


def load_wisdm(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    acc_files = _find_watch_files(data_dir, "accel")
    gyro_files = {_subject_id(p): p for p in _find_watch_files(data_dir, "gyro")}

    all_X, all_y, all_g = [], [], []
    for acc_path in acc_files:
        sid = _subject_id(acc_path)
        g_path = gyro_files.get(sid)
        X, y, _ = _windows_for_subject(acc_path, g_path)
        if not X:
            continue
        all_X.extend(X)
        all_y.extend(y)
        all_g.extend([sid] * len(y))
        log.info("Subject %s: %d windows", sid, len(y))

    if not all_X:
        raise FileNotFoundError(f"No WISDM watch windows in {data_dir}")
    return np.stack(all_X), np.array(all_y), np.array(all_g)


def download_wisdm(dest: Path) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    zip_path = dest / "wisdm.zip"
    if not zip_path.exists():
        log.info("Downloading WISDM from UCI (~250MB)...")
        r = requests.get(WISDM_URL, timeout=600, stream=True)
        r.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    extract = dest / "extracted"
    extract.mkdir(parents=True, exist_ok=True)
    if not any(extract.iterdir()):
        log.info("Extracting outer archive...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract)
    inner_zips = list(extract.glob("**/wisdm-dataset.zip"))
    data_root = extract / "wisdm-dataset"
    if inner_zips and not (data_root / "raw").exists():
        log.info("Extracting inner wisdm-dataset.zip (~300MB)...")
        with zipfile.ZipFile(inner_zips[0], "r") as zf:
            zf.extractall(extract)
    return extract


def run_loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> dict:
    logo = LeaveOneGroupOut()
    classes = sorted(set(y.tolist()))
    y_true, y_pred = [], []
    per_subj = []
    for tr, te in logo.split(X, y, groups):
        clf = RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        y_true.extend(y[te].tolist())
        y_pred.extend(pred.tolist())
        acc = accuracy_score(y[te], pred)
        walk_f1 = f1_score(y[te], pred, labels=["WALK"], average="macro", zero_division=0)
        per_subj.append(
            {"subject": str(groups[te[0]]), "accuracy": acc, "f1_walk": walk_f1, "n": len(te)}
        )
    y_true_arr = np.array(y_true)
    y_pred_arr = np.array(y_pred)
    acc = float(accuracy_score(y_true_arr, y_pred_arr))
    f1_walk = float(
        f1_score(y_true_arr, y_pred_arr, labels=["WALK"], average="macro", zero_division=0)
    )
    report = classification_report(y_true_arr, y_pred_arr, output_dict=True, zero_division=0)
    return {
        "accuracy": acc,
        "f1_walk": f1_walk,
        "classification_report": report,
        "per_subject": per_subj,
        "classes": classes,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(ROOT / "datasets" / "wisdm"))
    p.add_argument("--download", action="store_true")
    p.add_argument(
        "--output",
        default=str(ROOT / "models" / "motion_har_wisdm.joblib"),
    )
    p.add_argument(
        "--report",
        default=str(ROOT / "eval_results" / "wisdm_har_report.json"),
    )
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    data_dir = Path(args.data)
    if args.download or not any(data_dir.rglob("*.txt")):
        data_dir = download_wisdm(data_dir)

    X, y, groups = load_wisdm(data_dir)
    log.info("Total windows: %d, classes: %s", len(y), sorted(set(y)))

    loso = run_loso(X, y, groups)
    gate_passed = loso["accuracy"] >= GATE_ACCURACY and loso["f1_walk"] >= GATE_F1_WALK

    report = {
        "dataset": "WISDM smartwatch",
        "n_windows": int(len(y)),
        "n_subjects": int(len(set(groups.tolist()))),
        "feature_names": MOTION_FEATURE_NAMES,
        "window_sec": WINDOW_SEC,
        "gate": {
            "accuracy_min": GATE_ACCURACY,
            "f1_walk_min": GATE_F1_WALK,
            "accuracy": loso["accuracy"],
            "f1_walk": loso["f1_walk"],
            "passed": gate_passed,
        },
        "loso": loso,
    }

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    log.info("Report: %s (gate %s)", report_path, "PASS" if gate_passed else "FAIL")

    if args.eval_only:
        sys.exit(0 if gate_passed else 2)

    if not gate_passed:
        log.warning(
            "Gate FAIL (acc=%.3f, f1_walk=%.3f) — salvăm modelul pentru inferență Solid; "
            "auto-calibrarea per-user compensează parțial transferul WISDM→GW7.",
            loso["accuracy"],
            loso["f1_walk"],
        )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )
    clf.fit(X, y)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": clf,
            "classes": sorted(set(y.tolist())),
            "feature_names": MOTION_FEATURE_NAMES,
            "source": "WISDM_watch",
            "gate_passed": gate_passed,
            "loso_accuracy": loso["accuracy"],
            "loso_f1_walk": loso["f1_walk"],
        },
        out,
    )
    log.info("Model: %s", out)
    sys.exit(0 if gate_passed else 2)


if __name__ == "__main__":
    main()
