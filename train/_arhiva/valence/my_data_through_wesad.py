#!/usr/bin/env python3
"""Run MY real watch signal through the live WESAD valence model + explain it.

The live pipeline applies the WESAD model (trained on stress-vs-amusement on OTHER
people, lab clips, chest-clean signal) to ME (a new person, Galaxy wrist PPG, real
life). This shows what that mismatch looks like:
  1. how often the model is 'on the fence' (confidence near 0.5 = effectively a
     coin flip - why it looks random live), and
  2. SHAP: WHICH features drive its decision on my signal (usually HR/arousal,
     not valence - the disguised-arousal finding, on my own data).

Produces train/figs/my_wesad_confidence.png + train/figs/my_wesad_shap.png.
Run: ./venv/Scripts/python.exe train/results/valence/my_data_through_wesad.py
Needs: data/my_ppg_live.csv + models/valence_wesad.joblib
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.research.valence.features import (  # noqa: E402
    VALENCE_FEATURE_NAMES,
    extract_valence_feature_vector,
)

FS = 100
WIN_S = 20


def _my_windows() -> np.ndarray:
    df = pd.read_csv(ROOT / "data" / "my_ppg_live.csv")
    df = df[df["green"].notna()]
    g = df["green"].to_numpy(float)
    t = pd.to_datetime(df["time"], utc=True, format="ISO8601").astype("int64") // 1_000_000
    t = t.to_numpy()
    w = FS * WIN_S
    vecs = []
    for i in range(0, len(g) - w + 1, w // 2):
        seg_g = [int(round(x)) for x in g[i:i + w]]
        seg_t = [int(x) for x in t[i:i + w]]
        if 15 <= (seg_t[-1] - seg_t[0]) / 1000.0 <= 30:
            vec = extract_valence_feature_vector(seg_g, seg_t)
            if vec is not None and all(np.isfinite(vec)):
                vecs.append(vec)
    return np.asarray(vecs, float)


def main() -> None:
    import joblib
    b = joblib.load(ROOT / "models" / "valence_wesad.joblib")
    model = b["model"]
    fmean = np.asarray(b["feature_mean"], float)
    fstd = np.asarray(b["feature_std"], float)
    names = list(b["feature_names"])

    X = _my_windows()
    print(f"My live windows: {X.shape}")
    Xn = (X - fmean) / fstd
    proba = model.predict_proba(Xn)[:, 1]        # P(positive)

    # ---- 1. how often is it a coin flip? -------------------------------------
    on_fence = np.mean(np.abs(proba - 0.5) < 0.1)   # within 0.4-0.6
    print(f"\n=== Confidence on MY signal (WESAD model) ===")
    print(f"  mean p_positive: {proba.mean():.2f}  (0.5 = undecided)")
    print(f"  'on the fence' (0.4-0.6, basically a coin flip): {on_fence*100:.0f}% of windows")
    print(f"  -> this is why it looks random live: most of my real-life windows are "
          f"neither stress nor amusement, so the model has no confident answer.")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("(matplotlib indisponibil)")
        return

    out = ROOT / "train" / "figs"; out.mkdir(exist_ok=True)

    # confidence histogram
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.hist(proba, bins=30, color="#5dade2", edgecolor="white")
    ax.axvspan(0.4, 0.6, color="#e74c3c", alpha=0.15)
    ax.axvline(0.5, ls="--", c="#c0392b")
    ax.text(0.5, ax.get_ylim()[1]*0.9, "zona pacanea\n(0.4-0.6)",
            ha="center", color="#c0392b", fontsize=9)
    ax.set_xlabel("P(positive) - modelul WESAD pe semnalul MEU")
    ax.set_ylabel("nr ferestre")
    ax.set_title("Cat de des e modelul 'pe muchie' pe datele mele reale\n"
                 f"({on_fence*100:.0f}% din ferestre = practic aruncare cu banul)")
    fig.tight_layout()
    fig.savefig(out / "my_wesad_confidence.png", dpi=140)
    print(f"\nGrafic confidence: {out / 'my_wesad_confidence.png'}")

    # ---- 2. SHAP: which features drive the decision on my data? ----------------
    try:
        import shap
        # KernelExplainer works for SVC; use a small background sample for speed
        bg = shap.sample(Xn, min(50, len(Xn)), random_state=0)
        explainer = shap.KernelExplainer(
            lambda d: model.predict_proba(d)[:, 1], bg)
        sample = Xn[:80]                          # explain up to 80 windows
        sv = explainer.shap_values(sample, nsamples=100)

        plt.figure()
        shap.summary_plot(sv, sample, feature_names=names, show=False,
                          max_display=12, plot_size=(9, 6))
        plt.title("SHAP: ce features decid verdictul WESAD pe semnalul MEU",
                  fontsize=11)
        plt.tight_layout()
        plt.savefig(out / "my_wesad_shap.png", dpi=140, bbox_inches="tight")
        print(f"Grafic SHAP: {out / 'my_wesad_shap.png'}")

        # top features by mean |shap|
        imp = np.abs(sv).mean(0)
        order = np.argsort(-imp)[:6]
        print("\n=== Top 6 features care conduc decizia pe datele mele (SHAP) ===")
        for i in order:
            print(f"  {names[i]:18s}  importanta={imp[i]:.4f}")
        print("  -> daca domina hrv_mean_hr / features de HR = decizia e AROUSAL "
              "deghizat, nu valenta (confirma constatarea pe propriul semnal).")
    except Exception as e:  # noqa: BLE001
        print(f"\n(SHAP a esuat: {e})")


if __name__ == "__main__":
    main()
