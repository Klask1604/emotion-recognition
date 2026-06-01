#!/usr/bin/env python3
"""The TRANSFERABLE discomfort detector — ceiling test, offline (WESAD, no watch).

my_domain_shift.py showed (on the user's real 100 Hz watch signal) that only the
HRV/rate features transfer (|d|<0.5): hrv_rmssd, hrv_sdnn, hrv_lf, hrv_hf,
hrv_lf_hf, perfusion_index, reflection_idx. Everything morphological is shifted
beyond use on the wrist.

So the only discomfort detector that can survive on the watch is one built ONLY
from those transferable features. This script measures its CEILING on WESAD
(stress=discomfort vs amusement, LOSO balanced accuracy) — the best it could do
before per-subject validation. Then it checks the honest question: is the
transferable detector still mostly arousal (HR), or does it keep a discomfort
signal beyond heart rate?

Run: ./venv/Scripts/python.exe train/discomfort_transferable.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.legacy.valence_features import VALENCE_FEATURE_NAMES  # noqa: E402

# Features that transfer to the watch at 100 Hz (|d|<0.5 vs WESAD, measured on the
# user's own live PPG in my_domain_shift.py). The discomfort detector may use ONLY
# these if it is to work on the wrist.
TRANSFERABLE = [
    "hrv_rmssd", "hrv_sdnn", "hrv_lf", "hrv_hf", "hrv_lf_hf",
    "perfusion_index", "reflection_idx",
]
# Of those, hrv_mean_hr is NOT in the list (it shifted), but mean-HR-like rate is
# the arousal proxy; the transferable set is deliberately HRV-VARIABILITY, not rate.


def _loso(X, y, s):
    from sklearn.model_selection import LeaveOneGroupOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from sklearn.metrics import balanced_accuracy_score

    logo = LeaveOneGroupOut()
    acc = []
    for tr, te in logo.split(X, y, s):
        if len(np.unique(y[tr])) < 2:
            continue
        c = make_pipeline(StandardScaler(), SVC(kernel="rbf", class_weight="balanced"))
        c.fit(X[tr], y[tr])
        acc.append(balanced_accuracy_score(y[te], c.predict(X[te])))
    return float(np.mean(acc)) * 100 if acc else 0.0


def main() -> None:
    names = list(VALENCE_FEATURE_NAMES)
    d = np.load(ROOT / "data" / "wesad_features.npz", allow_pickle=True)
    X, y, s = d["X"], d["y"], d["subjects"]
    y_disc = 1 - y  # 1 = stress (discomfort), 0 = amusement

    idx_trans = [names.index(n) for n in TRANSFERABLE]
    idx_hr = [names.index("hrv_mean_hr")]

    print(f"WESAD discomfort (stress vs amusement) — n={len(y)}, "
          f"{len(np.unique(s))} subjects\n")

    print("=== Ceiling of the TRANSFERABLE discomfort detector ===")
    print(f"  all 33 features (NOT transferable)      : {_loso(X, y_disc, s):.0f}%")
    print(f"  transferable only ({len(TRANSFERABLE)} feat, works on watch): "
          f"{_loso(X[:, idx_trans], y_disc, s):.0f}%")
    print(f"  HR-only (pure arousal, the floor)       : {_loso(X[:, idx_hr], y_disc, s):.0f}%")

    # Is the transferable detector still mostly arousal, or real discomfort?
    # Add HR to the transferable set: if it barely moves, the transferable set
    # already captured the discomfort beyond HR.
    both = idx_trans + idx_hr
    print(f"\n=== Is the transferable detector arousal in disguise? ===")
    acc_trans = _loso(X[:, idx_trans], y_disc, s)
    acc_both = _loso(X[:, both], y_disc, s)
    print(f"  transferable only            : {acc_trans:.0f}%")
    print(f"  transferable + HR            : {acc_both:.0f}%  ({acc_both-acc_trans:+.0f} pp from adding HR)")
    if acc_trans - _loso(X[:, idx_hr], y_disc, s) >= 4:
        print("  -> the transferable set beats HR-only -> it carries REAL discomfort, "
              "not just arousal. Good: survives on the watch AND is more than rate.")
    else:
        print("  -> transferable set ~ HR-only -> it's mostly arousal. The discomfort "
              "specificity lived in morphology, which the watch loses.")

    # Per-feature: which transferable feature pulls the most weight.
    print(f"\n=== Single transferable feature, alone (which one carries it) ===")
    singles = []
    for n in TRANSFERABLE:
        a = _loso(X[:, [names.index(n)]], y_disc, s)
        singles.append((n, a))
    for n, a in sorted(singles, key=lambda r: -r[1]):
        print(f"  {n:18s}: {a:.0f}%")

    print("\nReading: the transferable-only number is the CEILING the watch detector "
          "can reach (before per-subject calibration). If it's well above HR-only, the "
          "discomfort axis is real AND wrist-viable. Validate this exact number on the "
          "user's own feedback labels when the watch is back.")


if __name__ == "__main__":
    main()
