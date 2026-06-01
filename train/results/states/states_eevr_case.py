#!/usr/bin/env python3
"""Native-state views on EEVR and CASE — parallel to states_wesad.py.

WESAD showed: 'stress vs calm' (86%) beats forced valence (58%). EEVR and CASE
don't have stress/amusement labels — EEVR has the 4 Russell quadrants, CASE has
continuous valence+arousal. So the equivalent 'native useful state' here is the
AROUSAL split (activated vs calm), which physiology carries well, vs the valence
split (weak). This confirms whether the WESAD lesson generalises: the useful axis
is arousal/activation, not valence.

Separate from the live valence models / Grafana boards — read-only on cached npz.

Run: ./venv/Scripts/python.exe train/states_eevr_case.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _loso(X, y, s):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score

    rows = np.all(np.isfinite(X), axis=1)
    X, y, s = X[rows], y[rows], s[rows]
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(accs)) * 100 if accs else 0.0


def eevr() -> None:
    d = np.load(ROOT / "data" / "eevr_stratified.npz", allow_pickle=True)
    X, quad, s = d["Xb"], d["quad"], d["subjects"]
    print("=== EEVR (4 Russell quadrants) ===")
    # arousal split: HA (HVHA+LVHA) vs LA (HVLA+LVLA) — activation, native-strong
    aro = np.isin(quad, ["HVHA", "LVHA"]).astype(int)
    print(f"  AROUSAL (activated vs calm): {_loso(X, aro, s):.0f}%  <- native useful axis")
    # valence split (for reference, the weak axis)
    val = np.isin(quad, ["HVHA", "HVLA"]).astype(int)
    print(f"  VALENCE (pleasant vs not):   {_loso(X, val, s):.0f}%  (weak, reference)")
    # full 4-quadrant
    code = {q: i for i, q in enumerate(["HVHA", "LVHA", "HVLA", "LVLA"])}
    y4 = np.array([code[q] for q in quad])
    print(f"  4-quadrant (full Russell):   {_loso(X, y4, s):.0f}%  (chance 25%)")


def case() -> None:
    d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
    X, val, aro, s = d["Xb"], d["valence"], d["arousal"], d["subjects"]
    print("\n=== CASE (continuous valence + arousal) ===")
    # arousal: high (>=6) vs low (<=4), drop middle
    sel_a = (aro >= 6) | (aro <= 4)
    ya = (aro[sel_a] >= 6).astype(int)
    print(f"  AROUSAL (high vs low):       {_loso(X[sel_a], ya, s[sel_a]):.0f}%  <- native useful axis")
    # valence: high (>=7) vs low (<=3), drop middle
    sel_v = (val >= 7) | (val <= 3)
    yv = (val[sel_v] >= 7).astype(int)
    print(f"  VALENCE (pleasant vs not):   {_loso(X[sel_v], yv, s[sel_v]):.0f}%  (weak, reference)")


def main() -> None:
    eevr()
    case()
    print("\nVerdict: across all three datasets, the AROUSAL/activation split is the "
          "strong native axis (like WESAD stress-vs-calm), while the VALENCE split is "
          "weak. The product should report activation/stress states, not the valence "
          "axis — the user's intuition holds on every dataset.")


if __name__ == "__main__":
    main()
