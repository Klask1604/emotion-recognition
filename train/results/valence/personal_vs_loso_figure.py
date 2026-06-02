#!/usr/bin/env python3
"""The thesis's answer figure: valence is weak ACROSS subjects, strong WITHIN one.

The honest finding is not "wrist PPG can't read valence" - it is "wrist PPG can't
read valence in a GENERAL (cross-subject) model, but a PERSONAL (per-user) model
recovers it." This recomputes both protocols on every dataset and plots them side
by side:

  LOSO     - train on N-1 subjects, test on a new one (a generic model shipped to
             a stranger). The regime where literature's 99% claims collapse.
  PERSONAL - train and test within each subject (5-fold), i.e. the model a wearable
             builds AFTER calibrating on its own user. The realistic product regime.

LogReg classifier, binary valence (negative vs positive), balanced accuracy.
Numbers (this run): WESAD 85->94, CASE 60->82, EEVR 56->72, EMOGNITION 50->69.
The gap IS the contribution: personalisation is what makes wrist valence usable.

Produces train/figs/personal_vs_loso.png and prints the table.
Run: ./venv/Scripts/python.exe train/results/valence/personal_vs_loso_figure.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))


def _model():
    from sklearn.linear_model import LogisticRegression
    return LogisticRegression(max_iter=1000, class_weight="balanced")


def _load(name):
    if name == "WESAD":
        d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
        return d["X"], d["y"].astype(int), d["subjects"]
    if name == "CASE":
        d = np.load(ROOT / "data" / "case_stratified.npz", allow_pickle=True)
        v = d["valence"]; sel = (v <= 3) | (v >= 7)
        return d["Xb"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel]
    if name == "EEVR":
        d = np.load(ROOT / "data" / "eevr_stratified.npz", allow_pickle=True)
        q = d["quad"]
        return d["Xb"], np.isin(q, ["HVHA", "HVLA"]).astype(int), d["subjects"]
    if name == "EMOGNITION":
        d = np.load(ROOT / "data" / "emognition_features.npz", allow_pickle=True)
        v = d["valence"]; sel = (v <= 3) | (v >= 7)
        return d["X"][sel], (v[sel] >= 7).astype(int), d["subjects"][sel]


def _loso(X, y, s):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    logo = LeaveOneGroupOut(); accs = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        m = make_pipeline(StandardScaler(), clone(_model())); m.fit(X[tr], y[tr])
        accs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
    return np.mean(accs) * 100


def _personal(X, y, s):
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.base import clone
    accs = []
    for u in np.unique(s):
        m = s == u; Xu, yu = X[m], y[m]
        if len(np.unique(yu)) < 2 or len(yu) < 10:
            continue
        nsplit = min(5, int(np.bincount(yu).min()))
        if nsplit < 2:
            continue
        skf = StratifiedKFold(n_splits=nsplit, shuffle=True, random_state=0)
        for tr, te in skf.split(Xu, yu):
            if len(np.unique(yu[tr])) < 2:
                continue
            c = make_pipeline(StandardScaler(), clone(_model())); c.fit(Xu[tr], yu[tr])
            accs.append(balanced_accuracy_score(yu[te], c.predict(Xu[te])))
    return np.mean(accs) * 100 if accs else float("nan")


def main() -> None:
    names = ["WESAD", "CASE", "EEVR", "EMOGNITION"]
    loso, pers = [], []
    print("Valence (negative vs positive), balanced accuracy %:")
    for n in names:
        X, y, s = _load(n)
        rows = np.all(np.isfinite(X), axis=1); X, y, s = X[rows], y[rows], s[rows]
        l, p = _loso(X, y, s), _personal(X, y, s)
        loso.append(l); pers.append(p)
        print(f"  {n:12s} LOSO={l:.0f}%  PERSONAL={p:.0f}%  (+{p-l:.0f})")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib indisponibil)")
        return

    x = np.arange(len(names)); w = 0.38
    fig, ax = plt.subplots(figsize=(9, 5.5))
    b1 = ax.bar(x - w / 2, loso, w, label="Model GENERAL (LOSO - testat pe strain)",
                color="#95a5a6")
    b2 = ax.bar(x + w / 2, pers, w, label="Model PERSONAL (calibrat pe utilizator)",
                color="#27ae60")
    ax.axhline(50, ls="--", c="#c0392b", lw=1)
    ax.text(len(names) - 0.6, 51, "sansa (50%)", color="#c0392b", fontsize=8)

    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"{b.get_height():.0f}", ha="center", fontsize=10, fontweight="bold")
    # draw the gap arrows
    for i, (l, p) in enumerate(zip(loso, pers)):
        ax.annotate(f"+{p-l:.0f}", xy=(i, (l + p) / 2), ha="center",
                    fontsize=9, color="#196f3d", fontweight="bold")

    ax.set_ylabel("Valenta - balanced accuracy %")
    ax.set_ylim(0, 100)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_title("Valenta din PPG: slaba pe STRAINI, recuperata PERSONAL\n"
                 "(acelaşi model LogReg - difera doar daca s-a calibrat pe utilizator)")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()

    out = ROOT / "train" / "figs"; out.mkdir(exist_ok=True)
    path = out / "personal_vs_loso.png"
    fig.savefig(path, dpi=140)
    print(f"\nGrafic salvat: {path}")
    print("Citire: modelul general (gri) cade spre sansa - de aceea cifrele de "
          "99% din literatura sunt suspecte. Modelul personal (verde), pe care un "
          "wearable il construieste dupa calibrarea pe utilizator, recupereaza "
          "valenta la 69-94%. Personalizarea ESTE solutia.")


if __name__ == "__main__":
    main()
