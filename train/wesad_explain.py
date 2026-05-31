#!/usr/bin/env python3
"""WHY does the WESAD valence model decide the way it does? — backtracking.

Two levels of interpretability for the thesis:

GLOBAL (which features carry valence across all subjects):
  - Cohen's d per feature (stress vs amusement separation) — raw signal strength,
    model-independent. Tells you which of the 33 features actually differ between
    negative and positive valence.
  - Permutation importance on the trained SVM (LOSO-style, on held-out data):
    how much balanced accuracy drops when each feature is shuffled. Model-dependent
    "what the SVM actually leans on".

PER-WINDOW (why THIS 20 s window was called positive/negative):
  - SHAP KernelExplainer on the trained RBF-SVM probability output. For a few
    concrete windows (a confident-positive, a confident-negative, an uncertain
    one) it shows each feature's signed push toward the prediction.

Reuses the exact extraction from train_valence_wesad.py (same windows, same
features, same normalize_ppg_window) so the explanation matches the live model.

Run:  ./venv/Scripts/python.exe train/wesad_explain.py
Cache: data/wesad_features.npz   Output: eval_results/wesad_explain.txt (+ PNGs)
"""

from __future__ import annotations

import glob
import pickle
import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 64
WIN_S = 20
LABEL_STRESS, LABEL_AMUSEMENT = 2, 3
CACHE = ROOT / "data" / "wesad_features.npz"
OUT = ROOT / "eval_results" / "wesad_explain.txt"
MODEL = ROOT / "models" / "valence_wesad.joblib"


def extract() -> dict:
    """Window WESAD wrist BVP into the same 33-feature vectors the model trains on."""
    if CACHE.exists():
        d = np.load(CACHE, allow_pickle=True)
        return {k: d[k] for k in d.files}
    X, y, subj = [], [], []
    for path in sorted(glob.glob(str(ROOT / "datasets/WESAD/S*/S*.pkl"))):
        sid = int(Path(path).stem[1:])
        d = pickle.load(open(path, "rb"), encoding="latin1")
        bvp = np.array(d["signal"]["wrist"]["BVP"]).flatten()
        lbl = np.array(d["label"])
        wb, wl = WIN_S * FS, WIN_S * 700
        for i in range(len(bvp) // wb):
            seg = bvp[i * wb:(i + 1) * wb]
            ls = lbl[i * wl:(i + 1) * wl]
            vals, cnts = np.unique(ls, return_counts=True)
            dom = vals[np.argmax(cnts)]
            if dom not in (LABEL_STRESS, LABEL_AMUSEMENT):
                continue
            vec = extract_valence_feature_vector(
                [int(round(x)) for x in seg],
                [int(j * 1000 / FS) for j in range(len(seg))],
            )
            if vec is None:
                continue
            X.append(vec); y.append(1 if dom == LABEL_AMUSEMENT else 0); subj.append(sid)
    out = dict(X=np.asarray(X, float), y=np.asarray(y, int), subjects=np.asarray(subj, int))
    CACHE.parent.mkdir(exist_ok=True)
    np.savez(CACHE, **out)
    return out


def cohens_d(x_pos: np.ndarray, x_neg: np.ndarray) -> np.ndarray:
    mu = x_pos.mean(0) - x_neg.mean(0)
    sd = np.sqrt((x_pos.var(0) + x_neg.var(0)) / 2) + 1e-9
    return mu / sd


def main() -> None:
    import joblib
    from sklearn.inspection import permutation_importance
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score
    import shap

    names = list(VALENCE_FEATURE_NAMES)
    d = extract()
    X, y, subj = d["X"], d["y"], d["subjects"]
    lines = []

    def out(s=""):
        print(s); lines.append(s)

    out(f"WESAD valence backtracking — n={len(y)} windows, "
        f"{len(np.unique(subj))} subjects, positive(amusement)={y.mean():.0%}")
    out("classes: 0=negative(stress)  1=positive(amusement)\n")

    # ---------- GLOBAL 1: Cohen's d (model-independent signal) ----------
    dpos = cohens_d(X[y == 1], X[y == 0])
    order = np.argsort(-np.abs(dpos))
    out("=== GLOBAL 1 — Cohen's d per feature (raw separation, no model) ===")
    out("  +d = higher in POSITIVE (amusement); -d = higher in NEGATIVE (stress)")
    for i in order[:12]:
        bar = "#" * int(min(abs(dpos[i]) * 10, 30))
        out(f"  {names[i]:18s} d={dpos[i]:+.2f}  {bar}")
    out("")

    # ---------- GLOBAL 2: permutation importance on the SVM (LOSO held-out) ----------
    out("=== GLOBAL 2 — permutation importance (what the SVM leans on) ===")
    out("  drop in balanced accuracy when each feature is shuffled (held-out folds)")
    logo = LeaveOneGroupOut()
    imp_acc = np.zeros(len(names))
    n_folds = 0
    for tr, te in logo.split(X, y, groups=subj):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        r = permutation_importance(clf, X[te], y[te], n_repeats=10,
                                   scoring="balanced_accuracy", random_state=0)
        imp_acc += r.importances_mean
        n_folds += 1
    imp_acc /= max(n_folds, 1)
    iorder = np.argsort(-imp_acc)
    for i in iorder[:12]:
        bar = "#" * int(min(max(imp_acc[i], 0) * 200, 30))
        out(f"  {names[i]:18s} drop={imp_acc[i]:+.4f}  {bar}")
    out("")

    # ---------- PER-WINDOW: SHAP on the production model ----------
    out("=== PER-WINDOW — SHAP (why specific windows were classified) ===")
    bundle = joblib.load(MODEL)
    model = bundle["model"]
    mean, std = bundle["feature_mean"], bundle["feature_std"]
    Xn = (X - mean) / std                       # same normalization as serving
    proba = model.predict_proba(Xn)[:, 1]       # P(positive)

    # pick three illustrative windows: confident-positive, confident-negative, uncertain
    idx_pos = int(np.argmax(proba))
    idx_neg = int(np.argmin(proba))
    idx_unc = int(np.argmin(np.abs(proba - 0.5)))

    # KernelExplainer on the probability output; small background for speed
    bg = shap.kmeans(Xn, 20)
    explainer = shap.KernelExplainer(lambda z: model.predict_proba(z)[:, 1], bg)

    for tag, idx in (("CONFIDENT POSITIVE", idx_pos),
                     ("CONFIDENT NEGATIVE", idx_neg),
                     ("UNCERTAIN (~0.5)", idx_unc)):
        sv = explainer.shap_values(Xn[idx:idx + 1], nsamples=200)[0]
        true = "positive" if y[idx] == 1 else "negative"
        out(f"\n  --- window #{idx} | P(positive)={proba[idx]:.2f} | "
            f"true={true} | subject S{subj[idx]} | {tag} ---")
        o = np.argsort(-np.abs(sv))
        out("    feature              SHAP    raw value   push")
        for i in o[:8]:
            push = "-> POSITIVE" if sv[i] > 0 else "-> negative"
            out(f"    {names[i]:18s} {sv[i]:+.3f}   {X[idx, i]:9.3f}   {push}")

    out("\nReading: SHAP>0 pushed the window toward POSITIVE(amusement), <0 toward "
        "NEGATIVE(stress). Compare top features here with the GLOBAL ranking above — "
        "they should agree (perfusion/vascular dominating) if the model is honest.")

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {OUT}")


if __name__ == "__main__":
    main()
