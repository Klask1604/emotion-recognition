#!/usr/bin/env python3
"""Random split vs LOSO across datasets — the inflation depends on signal quality.

Papers like WEARS (93%) and Nandini/EMOGNITION (99%, same Galaxy Watch) report
huge wearable-emotion accuracy using a RANDOM split: windows from the same subject
land in both train and test, so the model recognises the PERSON, not the emotion.
The honest metric is Leave-One-Subject-Out (LOSO).

This runs BOTH protocols on the SAME data for three datasets, so the gap is pure
method. The point is NOT "split is always bad" — it's that the inflation scales
with how weak the signal is:
  - WESAD stress-vs-calm: strong real signal -> small inflation, LOSO survives.
  - EEVR / CASE arousal (video clips): weak signal -> big inflation, LOSO collapses
    to chance, exactly like Quiroz's gait LOSO (AUC ~0.52).

Produces train/figs/split_inflation.png and prints the table.

Run: ./venv/Scripts/python.exe train/split_inflation.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _clf():
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    return make_pipeline(StandardScaler(),
                         SVC(kernel="rbf", class_weight="balanced"))


def _eval(X, y, s):
    """Return (random_split_acc, loso_acc) as balanced accuracy %."""
    from sklearn.model_selection import (
        LeaveOneGroupOut, StratifiedKFold, cross_val_score,
    )
    from sklearn.metrics import balanced_accuracy_score

    rows = np.all(np.isfinite(X), axis=1)
    X, y, s = X[rows], y[rows], s[rows]

    # random 5-fold on windows (subject leaks) = the inflated protocol
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    rand = cross_val_score(_clf(), X, y, cv=skf,
                           scoring="balanced_accuracy").mean() * 100

    # LOSO = honest
    logo = LeaveOneGroupOut()
    accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        c = _clf(); c.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    loso = float(np.mean(accs)) * 100 if accs else 0.0
    return rand, loso


def datasets():
    # WESAD: stress vs calm (the strong, real signal) ------------------------
    d = np.load(ROOT / "data" / "states_wesad.npz", allow_pickle=True)
    X, y, s = d["X"], d["y"], d["subjects"]
    keep = np.isin(y, [1, 2, 4])          # baseline, stress, meditation
    yield ("WESAD\nstres vs calm", X[keep], (y[keep] == 2).astype(int), s[keep])

    # EEVR: arousal (activated vs calm), weak video-clip signal --------------
    d = np.load(ROOT / "data" / "eevr_stratified.npz", allow_pickle=True)
    aro = np.isin(d["quad"], ["HVHA", "LVHA"]).astype(int)
    yield ("EEVR\narousal (clipuri)", d["Xb"], aro, d["subjects"])

    # CASE: arousal high vs low, weak video signal --------------------------
    d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
    a = d["arousal"]; sel = (a >= 6) | (a <= 4)
    yield ("CASE\narousal (clipuri)", d["Xb"][sel], (a[sel] >= 6).astype(int),
           d["subjects"][sel])


def main() -> None:
    rows = []
    for name, X, y, s in datasets():
        rand, loso = _eval(X, y, s)
        rows.append((name, rand, loso))
        flat = name.replace("\n", " ")
        print(f"{flat:24s}  random={rand:4.0f}%   LOSO={loso:4.0f}%   "
              f"umflare=+{rand - loso:.0f}pp")

    # ---- plot ----------------------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib indisponibil - sar peste grafic)")
        return

    names = [r[0] for r in rows]
    rand = [r[1] for r in rows]
    loso = [r[2] for r in rows]
    x = np.arange(len(names)); w = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    b1 = ax.bar(x - w / 2, rand, w, label="Split aleatoriu (subiectul se scurge)",
                color="#d98880")
    b2 = ax.bar(x + w / 2, loso, w, label="LOSO (cinstit, cross-subiect)",
                color="#5dade2")
    ax.axhline(50, ls="--", c="gray", lw=1)
    ax.text(len(names) - 0.5, 51, "sansa (50%)", color="gray", fontsize=8)

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=9)

    ax.set_ylabel("Balanced accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9)
    ax.set_title("Cat umfla split-ul aleatoriu depinde de cat de real e semnalul")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    out = ROOT / "train" / "figs"; out.mkdir(exist_ok=True)
    path = out / "split_inflation.png"
    fig.savefig(path, dpi=140)
    print(f"\nGrafic salvat: {path}")
    print("\nCitire: pe WESAD (semnal real de stres) LOSO supravietuieste -> "
          "umflarea e mica. Pe EEVR/CASE (arousal din clipuri = semnal slab) "
          "LOSO cade la sansa -> splitul aleatoriu umfla mult, exact ca la "
          "Quiroz (gait LOSO AUC~0.52) si Nandini/WEARS care raporteaza 93-99%.")


if __name__ == "__main__":
    main()
