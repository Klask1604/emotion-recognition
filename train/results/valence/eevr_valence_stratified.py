#!/usr/bin/env python3
"""EEVR valence, AROUSAL-STRATIFIED — the corrected version of eevr_test.py.

The flaw in eevr_test.py: it collapses the quadrant label to `valence = 1 if 'HV'
else 0`, pooling LVLA+LVHA vs HVLA+HVHA. That mixes the arousal axis into the
comparison and dilutes the valence signal to chance (~51%). WESAD reaches ~82%
precisely because it holds arousal CONSTANT (stress vs amusement, both
high-arousal) so the classifier is forced onto valence-bearing signal.

This script applies the same trick to EEVR. The clip label encodes the full
Russell quadrant in its middle token, e.g. `V4_HVHA_Speed Flying` -> HVHA
(High Valence, High Arousal). The four quadrants are:
    HVHA  high valence, high arousal   (puppies, coaster, speed flying)
    LVHA  low  valence, high arousal   (kidnapped, zombie, jailbreak)
    HVLA  high valence, low  arousal   (sunrise, caribbean, ocean road)
    LVLA  low  valence, low  arousal   (the displaced, abandoned building)

Three LOSO comparisons, balanced accuracy:
  1. HVHA vs LVHA  -> both HIGH-arousal, differ only in valence. THE NUMBER.
  2. HVLA vs LVLA  -> both LOW-arousal valence contrast (weaker expected).
  3. all-HV vs all-LV -> reproduces the eevr_test 51% flaw for a delta.

Two feature extractors are compared:
  (A) per-family path (FD+morph+HRV+vascular), the eevr_test lineage.
  (B) the shared extract_valence_feature_vector + normalize_ppg_window, i.e. the
      exact extractor the live watch channel uses (median+MAD scale-invariance).

Empatica E4 BVP = 64 Hz fixed. The `Time` column is fractional-second aggregated,
NOT a usable per-sample timestamp (it implies a false ~7700 Hz), so FS is hard-coded.

Run:  ./venv/Scripts/python.exe train/eevr_valence_stratified.py
Cache: data/eevr_stratified.npz  (per-window features + quadrant tags, reused by Exp 3)
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_ppg_fd import extract_valence_fd_features  # noqa: E402
from affectus.legacy.valence_ppg_morph import (  # noqa: E402
    MORPH_FEATURE_NAMES,
    extract_morph_features,
)
from affectus.legacy.ppg_hrv_features import (  # noqa: E402
    HRV_FEATURE_NAMES,
    extract_hrv_features,
)
from affectus.legacy.ppg_vascular_features import (  # noqa: E402
    VASC_FEATURE_NAMES,
    extract_vascular_features,
)
from affectus.legacy.valence_features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 64                     # Empatica E4 BVP, fixed (Time column is unusable)
WIN_S, STEP_S = 20, 10      # parity with WESAD windowing
FD_KEYS = ["bf_n", "fhf_n", "shf_n", "fhf_bf", "shf_bf", "shf_fhf"]
QUADRANTS = ("HVHA", "LVHA", "HVLA", "LVLA")
CACHE = ROOT / "data" / "eevr_stratified.npz"
PPG_CSV = ROOT / "datasets/EEVR/Physiological_Data/Raw_PPG.csv"

# Per-family feature order (extractor A) — matches eevr_test.py lineage.
FAMILY_NAMES = (
    FD_KEYS + list(MORPH_FEATURE_NAMES) + list(HRV_FEATURE_NAMES) + list(VASC_FEATURE_NAMES)
)
_N_FD_MO = len(FD_KEYS) + len(MORPH_FEATURE_NAMES)
_N_HRV = len(HRV_FEATURE_NAMES)
FAMILY_VASC_IDX = list(range(_N_FD_MO + _N_HRV, len(FAMILY_NAMES)))   # vascular slice
FAMILY_NONVASC_IDX = list(range(_N_FD_MO + _N_HRV))                   # fd+morph+hrv slice


def _quadrant(label: str) -> str | None:
    """Middle token of `V<n>_<QUAD>_<clip>`; None for Baseline / unparseable."""
    parts = str(label).split("_")
    if len(parts) >= 3 and parts[1] in QUADRANTS:
        return parts[1]
    return None


def _to_int_ppg(seg: np.ndarray) -> list[int]:
    """Scale float BVP (~0.07 units) to int-ish, matching eevr_test.feats()."""
    return [int(round(x * 1000)) for x in seg]


def feats_family(seg: np.ndarray) -> list[float] | None:
    """Extractor A: per-family path, the eevr_test lineage (no normalize_ppg_window)."""
    ts = [int(j * 1000 / FS) for j in range(len(seg))]
    g = _to_int_ppg(seg)
    # HR for harmonic anchoring (same band-pass peak count eevr_test uses).
    from scipy.signal import butter, filtfilt, find_peaks

    s = seg - seg.mean()
    b, a = butter(2, [0.5 / (FS / 2), 4 / (FS / 2)], btype="band")
    f = filtfilt(b, a, s)
    pk, _ = find_peaks(f, distance=int(0.4 * FS))
    hr = len(pk) / (len(seg) / FS) * 60 if len(pk) > 1 else 0.0
    if hr <= 0:
        return None
    fd = extract_valence_fd_features(g, ts, hr_bpm=hr)
    mo = extract_morph_features(g, ts)
    hv = extract_hrv_features(g, ts)
    vc = extract_vascular_features(g, ts)
    if not (fd.valid and mo.valid and hv.valid and vc.valid):
        return None
    return (
        [fd.as_dict()[k] for k in FD_KEYS]
        + [mo.as_dict()[k] for k in MORPH_FEATURE_NAMES]
        + [hv.as_dict()[k] for k in HRV_FEATURE_NAMES]
        + [vc.as_dict()[k] for k in VASC_FEATURE_NAMES]
    )


def feats_shared(seg: np.ndarray) -> list[float] | None:
    """Extractor B: the live watch extractor (median+MAD scale-invariance)."""
    ts = [int(j * 1000 / FS) for j in range(len(seg))]
    g = _to_int_ppg(seg)
    return extract_valence_feature_vector(g, ts)


def extract_all() -> dict:
    """Window every clip per (subject, label), run both extractors, tag quadrant."""
    print(f"Loading EEVR Raw_PPG ({PPG_CSV.stat().st_size / 1e6:.0f} MB)...")
    df = pd.read_csv(PPG_CSV, usecols=["PPG", "Label", "Participant ID"])
    df["quad"] = df["Label"].map(_quadrant)
    df = df[df["quad"].notna()]
    print(f"  windows over {df['Participant ID'].nunique()} subjects, "
          f"quadrants={sorted(df['quad'].unique())}")

    Xa, Xb, quad, subj = [], [], [], []
    w, st = WIN_S * FS, STEP_S * FS
    for (pid, label), grp in df.groupby(["Participant ID", "Label"]):
        sig = grp["PPG"].to_numpy(float)
        if len(sig) < w:
            continue
        q = grp["quad"].iloc[0]
        for i in range(0, len(sig) - w + 1, st):
            seg = sig[i:i + w]
            fa = feats_family(seg)
            fb = feats_shared(seg)
            if fa is None or fb is None:
                continue
            Xa.append(fa); Xb.append(fb); quad.append(q); subj.append(int(pid))
    out = dict(
        Xa=np.asarray(Xa, float),
        Xb=np.asarray(Xb, float),
        quad=np.asarray(quad),
        subjects=np.asarray(subj, int),
    )
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    print(f"  cached {len(quad)} windows -> {CACHE.name}")
    return out


def load() -> dict:
    if CACHE.exists():
        print(f"Loading cached {CACHE.name}")
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    return extract_all()


def loso(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> tuple[float, int]:
    """LOSO balanced accuracy; skips folds that are single-class. Returns (%, n_folds)."""
    rows = np.all(np.isfinite(X), axis=1)
    X, y, groups = X[rows], y[rows], groups[rows]
    logo = LeaveOneGroupOut()
    scores = []
    for tr, te in logo.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 1:
            continue
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        scores.append(balanced_accuracy_score(y[te], clf.predict(X[te])))
    return (float(np.mean(scores)) * 100 if scores else 0.0), len(scores)


def comparison(d: dict, pos: tuple[str, ...], neg: tuple[str, ...], title: str):
    """Run one valence contrast (pos quadrants=1, neg=0) for both extractors + ablations."""
    quad, subj = d["quad"], d["subjects"]
    mask = np.isin(quad, pos + neg)
    y = np.isin(quad[mask], pos).astype(int)
    g = subj[mask]
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    n_subj = len(np.unique(g))
    print(f"\n=== {title} ===")
    print(f"  windows: {len(y)}  (pos={n_pos}, neg={n_neg})  subjects={n_subj}")

    for tag, X in (("family", d["Xa"][mask]), ("shared", d["Xb"][mask])):
        acc, nf = loso(X, y, g)
        print(f"  [{tag}] ALL features   : {acc:5.1f}%   ({nf} folds)")
        if tag == "family":
            av, _ = loso(X[:, FAMILY_VASC_IDX], y, g)
            an, _ = loso(X[:, FAMILY_NONVASC_IDX], y, g)
            print(f"  [{tag}] vascular only  : {av:5.1f}%")
            print(f"  [{tag}] fd+morph+hrv   : {an:5.1f}%")


def main():
    d = load()
    quad = d["quad"]
    print("\nwindow counts per quadrant:",
          {q: int((quad == q).sum()) for q in QUADRANTS})

    # 1. PRIMARY: arousal held HIGH, valence contrast (the WESAD trick on EEVR).
    comparison(d, pos=("HVHA",), neg=("LVHA",),
               title="HVHA vs LVHA  [high-arousal, valence isolated] <- PRIMARY")
    # 2. Arousal held LOW, valence contrast.
    comparison(d, pos=("HVLA",), neg=("LVLA",),
               title="HVLA vs LVLA  [low-arousal, valence isolated]")
    # 3. The flawed pooling that eevr_test does -> expect ~chance.
    comparison(d, pos=("HVHA", "HVLA"), neg=("LVHA", "LVLA"),
               title="all-HV vs all-LV  [arousal MIXED — reproduces ~51% flaw]")

    print("\nDone. PRIMARY = HVHA-vs-LVHA. Target was 65%.")


if __name__ == "__main__":
    main()
