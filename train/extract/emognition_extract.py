#!/usr/bin/env python3
"""Extract the 33-feature PPG vectors from EMOGNITION's Samsung Galaxy Watch BVP.

EMOGNITION (Saganowski et al., 2022, Scientific Data) records the SAME Galaxy Watch
PPG that Biofizic uses, on 43 subjects, across 11 emotion conditions with SAM
valence/arousal self-reports. This makes it the cleanest external check of the
graded-valence finding on the exact product sensor.

Per subject folder N/ :
  N_<EMOTION>_STIMULUS_SAMSUNG_WATCH.json -> {"BVPRaw": [[ts_str, value], ...]}  (~20 Hz, 120 s)
  N_QUESTIONNAIRES.json -> questionnaires[i] = {"movie": EMOTION, "sam": {VALENCE, AROUSAL}}

We window the stimulus BVP (20 s, 10 s step), extract the shared 33-feature vector,
and label each window with that emotion's SAM valence + arousal. Cached to
data/emognition_features.npz so the 16 GB raw can be deleted afterwards.

Run: ./venv/Scripts/python.exe train/extract/emognition_extract.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from affectus.research.valence.features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

DATA = ROOT / "datasets" / "emognition"
OUT = ROOT / "data" / "emognition_features.npz"
WIN_S = 20
STEP_S = 10
EMOTIONS = ["AMUSEMENT", "ANGER", "AWE", "BASELINE", "DISGUST", "ENTHUSIASM",
            "FEAR", "LIKING", "NEUTRAL", "SADNESS", "SURPRISE"]


def _labels(subject_dir: Path) -> dict[str, tuple[int, int]]:
    """emotion -> (valence, arousal) from the subject's questionnaire."""
    qf = subject_dir / f"{subject_dir.name}_QUESTIONNAIRES.json"
    if not qf.exists():
        return {}
    q = json.load(open(qf))["questionnaires"]
    out = {}
    for item in q:
        sam = item.get("sam") or {}
        v, a = sam.get("VALENCE"), sam.get("AROUSAL")
        if item.get("movie") and v is not None and a is not None:
            out[item["movie"]] = (int(v), int(a))
    return out


def _bvp_windows(bvp_file: Path):
    """Yield (green_ints, ts_ms) windows from a Samsung Watch BVPRaw file."""
    d = json.load(open(bvp_file))
    raw = d.get("BVPRaw") or []
    if len(raw) < WIN_S * 15:
        return
    ts = pd.to_datetime([r[0] for r in raw], format="%Y-%m-%dT%H:%M:%S:%f")
    t_ms = (ts.astype("int64") // 1_000_000).to_numpy()
    vals = np.array([int(r[1]) for r in raw], float)
    # approximate sampling rate -> samples per window/step
    dur_s = (t_ms[-1] - t_ms[0]) / 1000.0
    if dur_s <= 0:
        return
    fs = len(raw) / dur_s
    w, step = int(WIN_S * fs), int(STEP_S * fs)
    if w < 32:
        return
    for i in range(0, len(vals) - w + 1, max(step, 1)):
        seg = vals[i:i + w]
        seg_t = t_ms[i:i + w]
        yield [int(round(x)) for x in seg], [int(x) for x in seg_t]


def main() -> None:
    subjects = sorted([p for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()],
                      key=lambda p: int(p.name))
    print(f"EMOGNITION: {len(subjects)} subjects")

    X, valence, arousal, subj, emo = [], [], [], [], []
    for sd in subjects:
        labels = _labels(sd)
        if not labels:
            continue
        n_sub = 0
        for emotion in EMOTIONS:
            if emotion not in labels:
                continue
            bvp_file = sd / f"{sd.name}_{emotion}_STIMULUS_SAMSUNG_WATCH.json"
            if not bvp_file.exists():
                continue
            v, a = labels[emotion]
            for green, ts in _bvp_windows(bvp_file):
                vec = extract_valence_feature_vector(green, ts)
                if vec is None or not all(np.isfinite(vec)):
                    continue
                X.append(vec); valence.append(v); arousal.append(a)
                subj.append(int(sd.name)); emo.append(emotion); n_sub += 1
        print(f"  subject {sd.name}: {n_sub} windows")

    X = np.asarray(X, float)
    print(f"\nTotal: {X.shape} windows, {len(set(subj))} subjects")
    OUT.parent.mkdir(exist_ok=True)
    np.savez(OUT, X=X, valence=np.asarray(valence, int),
             arousal=np.asarray(arousal, int), subjects=np.asarray(subj, int),
             emotion=np.asarray(emo), feature_names=np.asarray(list(VALENCE_FEATURE_NAMES)))
    print(f"Saved: {OUT}  ({OUT.stat().st_size // 1024} KB)")
    print("Raw EMOGNITION (16 GB) can now be deleted - features are cached.")


if __name__ == "__main__":
    main()
