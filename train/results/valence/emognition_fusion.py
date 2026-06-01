#!/usr/bin/env python3
"""Does EEG (FAA) or multimodal fusion beat PPG-alone for valence on EMOGNITION?

PPG-alone valence on EMOGNITION was chance (~51% LOSO). The literature (EEVR, the
Biofizic options review) says the real lift comes from EEG frontal alpha asymmetry
(FAA - the one direct valence channel) or from fusing PPG+EDA+EEG, not from PPG
alone. EMOGNITION has all three on the SAME subjects, so we can test it directly,
on the exact product watch, without buying a Muse.

Per emotion window we build:
  - PPG  : the 33 shared features (already cached in emognition_features.npz)
  - FAA  : ln(Alpha_AF8) - ln(Alpha_AF7) and a few EEG-band asymmetries (Muse)
  - EDA  : tonic mean/std + phasic activity (Empatica)
labelled with that emotion's SAM valence. Negative = stress-type emotions, etc.
Compares LOSO balanced accuracy: PPG, FAA-only, EDA-only, and PPG+FAA+EDA fused.

Run: ./venv/Scripts/python.exe train/results/valence/emognition_fusion.py
Needs: datasets/emognition/ (raw) - reads Muse + Empatica per window.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

DATA = ROOT / "datasets" / "emognition"
EMOTIONS = ["AMUSEMENT", "ANGER", "AWE", "BASELINE", "DISGUST", "ENTHUSIASM",
            "FEAR", "LIKING", "NEUTRAL", "SADNESS", "SURPRISE"]
NEG = {"ANGER", "DISGUST", "FEAR", "SADNESS"}
POS = {"AMUSEMENT", "ENTHUSIASM", "AWE", "LIKING"}


def _labels(sd: Path) -> dict[str, int]:
    qf = sd / f"{sd.name}_QUESTIONNAIRES.json"
    if not qf.exists():
        return {}
    q = json.load(open(qf))["questionnaires"]
    return {it["movie"]: int(it["sam"]["VALENCE"])
            for it in q if it.get("movie") and it.get("sam", {}).get("VALENCE") is not None}


def _faa_features(sd: Path, emotion: str) -> list[float] | None:
    """Frontal-alpha-asymmetry + band asymmetries from the Muse stimulus file."""
    mf = sd / f"{sd.name}_{emotion}_STIMULUS_MUSE.json"
    if not mf.exists():
        return None
    d = json.load(open(mf))

    def _mean(key):
        v = np.asarray(d.get(key, []), float)
        v = v[np.isfinite(v) & (v > 0)]
        return float(np.mean(v)) if len(v) else np.nan

    a7, a8 = _mean("Alpha_AF7"), _mean("Alpha_AF8")
    if not (np.isfinite(a7) and np.isfinite(a8)):
        return None
    faa_frontal = np.log(a8) - np.log(a7)          # >0 left-active = positive
    # band asymmetries (right - left) as extra valence-ish features
    feats = [faa_frontal]
    for band in ("Beta", "Gamma", "Theta"):
        l, r = _mean(f"{band}_AF7"), _mean(f"{band}_AF8")
        feats.append(np.log(r) - np.log(l) if (np.isfinite(l) and np.isfinite(r)
                                               and l > 0 and r > 0) else 0.0)
    # temporal-pair alpha asymmetry too (TP9/TP10)
    t9, t10 = _mean("Alpha_TP9"), _mean("Alpha_TP10")
    feats.append(np.log(t10) - np.log(t9) if (np.isfinite(t9) and np.isfinite(t10)
                                              and t9 > 0 and t10 > 0) else 0.0)
    return feats if all(np.isfinite(feats)) else None


def _eda_features(sd: Path, emotion: str) -> list[float] | None:
    """Tonic + phasic EDA summary from the Empatica stimulus file."""
    ef = sd / f"{sd.name}_{emotion}_STIMULUS_EMPATICA.json"
    if not ef.exists():
        return None
    d = json.load(open(ef))
    eda = d.get("EDA", [])
    if len(eda) < 5:
        return None
    v = np.asarray([x[1] for x in eda], float)
    v = v[np.isfinite(v)]
    if len(v) < 5:
        return None
    diff = np.diff(v)
    return [float(v.mean()), float(v.std()),
            float(np.mean(np.abs(diff))),           # phasic activity
            float((diff > 0).mean())]               # fraction rising


def main() -> None:
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import balanced_accuracy_score

    # PPG features already cached - but we need ONE vector per (subject, emotion)
    # to align with the single FAA/EDA value per stimulus. Average PPG windows.
    cache = np.load(ROOT / "data" / "emognition_features.npz", allow_pickle=True)
    Xp, vc, sc, ec = cache["X"], cache["valence"], cache["subjects"], cache["emotion"]

    rows_ppg = {}
    for x, v, s, e in zip(Xp, vc, sc, ec):
        rows_ppg.setdefault((int(s), str(e)), []).append(x)
    ppg_mean = {k: np.nanmean(np.asarray(v), axis=0) for k, v in rows_ppg.items()}

    subjects = sorted([p for p in DATA.iterdir() if p.is_dir() and p.name.isdigit()],
                      key=lambda p: int(p.name))
    print(f"Aligning PPG+FAA+EDA per (subject, emotion) over {len(subjects)} subjects...")

    PPG, FAA, EDA, Y, S = [], [], [], [], []
    for sd in subjects:
        labels = _labels(sd)
        sid = int(sd.name)
        for emo in EMOTIONS:
            if emo not in labels or emo not in (NEG | POS):
                continue
            key = (sid, emo)
            if key not in ppg_mean:
                continue
            faa = _faa_features(sd, emo)
            eda = _eda_features(sd, emo)
            if faa is None or eda is None:
                continue
            PPG.append(ppg_mean[key]); FAA.append(faa); EDA.append(eda)
            Y.append(1 if emo in POS else 0); S.append(sid)

    PPG, FAA, EDA = np.asarray(PPG), np.asarray(FAA), np.asarray(EDA)
    Y, S = np.asarray(Y), np.asarray(S)
    print(f"Aligned: {len(Y)} samples ({Y.sum()} positive / {(Y==0).sum()} negative), "
          f"{len(set(S))} subjects\n")

    def loso(X):
        logo = LeaveOneGroupOut(); accs = []
        for tr, te in logo.split(X, Y, S):
            if len(np.unique(Y[tr])) < 2 or len(np.unique(Y[te])) < 2:
                continue
            c = make_pipeline(SimpleImputer(strategy="mean"), StandardScaler(),
                              SVC(kernel="rbf", class_weight="balanced"))
            c.fit(X[tr], Y[tr]); accs.append(balanced_accuracy_score(Y[te], c.predict(X[te])))
        return np.mean(accs) * 100 if accs else 0.0

    print("=== Valence (negative vs positive emotions), LOSO balanced accuracy ===")
    print(f"  PPG alone (33 feat):      {loso(PPG):.0f}%")
    print(f"  FAA / EEG alone (6 feat): {loso(FAA):.0f}%   <- the direct valence channel")
    print(f"  EDA alone (4 feat):       {loso(EDA):.0f}%")
    print(f"  PPG + EDA:                {loso(np.hstack([PPG, EDA])):.0f}%")
    print(f"  PPG + FAA:                {loso(np.hstack([PPG, FAA])):.0f}%")
    print(f"  PPG + FAA + EDA (fused):  {loso(np.hstack([PPG, FAA, EDA])):.0f}%   <- full multimodal")

    print("\nReading: if FAA or fusion clearly beats PPG-alone, EEG adds a real "
          "valence channel the wrist can't provide - the paper's main recommendation. "
          "If they all stay near chance, even multimodal valence collapses on these "
          "mild film-clip stimuli (the honest graded-valence result holds).")


if __name__ == "__main__":
    main()
