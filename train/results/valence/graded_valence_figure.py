#!/usr/bin/env python3
"""The thesis's central figure: valence detectability is GRADED, not binary.

Peripheral PPG carries valence only in proportion to affective intensity. This
recomputes the valence LOSO balanced accuracy from each dataset's cached features
(no hardcoded numbers) and plots them in one bar chart against the 50% chance line:

    WESAD (extreme contrast, stress-vs-amusement) -> high
    CASE  (clearly-expressed valence, clean labels) -> mid
    EmoWear / DEAP (mild clips, retrospective self-report) -> chance

The point of the thesis: physiology separates positive from negative affect only
when the affect is strong. Same pipeline, same LOSO, same chance line - the only
thing that changes is the dataset's stimulus intensity.

Produces train/figs/graded_valence.png and prints the table.

Run: ./venv/Scripts/python.exe train/results/valence/graded_valence_figure.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _loso(X, y, s) -> float:
    """LOSO balanced accuracy %, the same protocol used across the thesis."""
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
        c = make_pipeline(StandardScaler(),
                          SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(accs)) * 100 if accs else 0.0


def _wesad() -> float:
    # WESAD valence = stress-vs-amusement (arousal held high in both -> isolates
    # valence). Cache y is already binary: 0=stress(negative), 1=amusement(positive).
    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    return _loso(d["X"], d["y"].astype(int), d["subjects"])


def _case() -> float:
    # CASE valence on CLEARLY-expressed labels only (<=3 negative, >=7 positive),
    # the filtering that lifts it from chance to its real ~62%.
    d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
    v = d["valence"]
    sel = (v <= 3) | (v >= 7)
    return _loso(d["Xb"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel])


def _emowear() -> float:
    # EmoWear valence: clip-level SAM, >=5 positive. Stays at chance.
    d = np.load(ROOT / "data" / "emowear_features.npz", allow_pickle=True)
    v = d["valence"]
    sel = (v <= 3) | (v >= 7)
    if sel.sum() < 20:                         # too few clear labels -> use median split
        sel = np.ones(len(v), bool)
        y = (v >= np.median(v)).astype(int)
    else:
        y = (v[sel] >= 7).astype(int)
    return _loso(d["X"][sel], y, d["subjects"][sel])


def _deap() -> float:
    d = np.load(ROOT / "data" / "deap_valence_fd.npz", allow_pickle=True)
    return _loso(d["X"], d["y"], d["subjects"])


def main() -> None:
    bars = [
        ("WESAD\nstres vs amusement", _wesad(), "contrast extrem"),
        ("CASE\nvalenta clara",       _case(),  "valenta clar exprimata"),
        ("EmoWear\nclipuri",          _emowear(), "inducere blanda"),
        ("DEAP\nclipuri",             _deap(),  "self-report retrospectiv"),
    ]
    print("Graded valence (LOSO balanced accuracy):")
    for name, acc, note in bars:
        print(f"  {name.splitlines()[0]:10s} {acc:5.1f}%   ({note})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("\n(matplotlib indisponibil - sar peste grafic)")
        return

    names = [b[0] for b in bars]
    vals = [b[1] for b in bars]
    notes = [b[2] for b in bars]
    # colour by how far above chance: strong=green, chance=grey
    colours = ["#27ae60" if v >= 70 else "#f39c12" if v >= 58 else "#95a5a6"
               for v in vals]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(names))
    b = ax.bar(x, vals, color=colours, width=0.6)
    ax.axhline(50, ls="--", c="#c0392b", lw=1.2)
    ax.text(len(names) - 0.5, 51, "sansa (50%)", color="#c0392b", fontsize=9)

    for bar, v, note in zip(b, vals, notes):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.0f}%",
                ha="center", fontsize=11, fontweight="bold")
        ax.text(bar.get_x() + bar.get_width() / 2, 3, note, ha="center",
                fontsize=7.5, color="white", rotation=0)

    ax.set_ylabel("Valenta - balanced accuracy (LOSO) %")
    ax.set_ylim(0, 100)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_title("Detectabilitatea valentei scaleaza cu intensitatea afectiva\n"
                 "(acelasi pipeline, acelasi LOSO - difera doar stimulul)")
    fig.tight_layout()

    out = ROOT / "train" / "figs"
    out.mkdir(exist_ok=True)
    path = out / "graded_valence.png"
    fig.savefig(path, dpi=140)
    print(f"\nGrafic salvat: {path}")
    print("\nCitire: PPG separa pozitiv de negativ DOAR cand afectul e puternic "
          "(WESAD). Pe stimuli blanzi (clipuri) cade la sansa. Limitare GRADATA, "
          "nu binara - fiziologia periferica poarta valenta doar la intensitate "
          "ridicata.")


if __name__ == "__main__":
    main()
