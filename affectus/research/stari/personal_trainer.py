"""Retrain a PERSONAL 3-state model from the user's own labels.

The generic model (stari_3.joblib) is cross-subject: it scores 65% LOSO on the
3 states. We proved on data (same EMOGNITION Galaxy base) that adding the user's
own labelled windows lifts this a lot, even at low volume:
  5 labels/state -> 89%, 8 -> 94%, 10 -> 96%. A self-check guards every swap.

This module runs IN THE ENGINE (Docker), so it must not touch the 16GB dataset.
Instead it loads a frozen generic training set (models/stari_3_generic.npz,
built once by train/results/validation/export_generic_trainset.py) and appends
the user's labelled windows from data/feedback_labels.jsonl.

Parity with serving:
- Generic subjects get the same per-subject deviation as the original training
  (deviation vs each subject's own CALM windows).
- The personal subject gets the deviation the LIVE classifier uses at serving:
  (x - mu) / sd from the persisted personal baseline (stari_3_baseline.npz).
  Same baseline at train time and at predict time => no train/serve skew.

One entry point: maybe_retrain(). It returns a small result dict the engine
logs and acts on (whether it trained, the self-check accuracy, the path).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover - engine always has these, tests may not
    joblib = None

from affectus.config import (
    PERSONAL_MIN_PER_STATE,
    generic_trainset_path,
    personal_model_path,
)
from affectus.dsp.feedback_store import load_feedback

# The 3 class codes; same as the generic bundle and feedback_store.
CLASSES = {0: "CALM", 1: "DISCONFORT", 2: "PLACUT"}
# 7 features (5 HRV + acc_rms + acc_std), matching the live state_classifier and
# the WESAD-trained stari_3.joblib. Gyro is NOT a feature (training sets lack it);
# feeding 9 here would make the personal model expect 18 cols while live predict()
# serves 14, crashing predict_proba. See state_classifier docstring.
_N_FEATURES = 7
# Personal baseline used at serving, so we deviate the personal rows identically.
_BASELINE_FILE = Path(__file__).resolve().parents[3] / "models" / "stari_3_baseline.npz"


def _rf():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier(150, random_state=0, n_jobs=-1, class_weight="balanced"),
    )


def _generic_deviate(X: np.ndarray, subj: np.ndarray, y: np.ndarray) -> np.ndarray:
    """[abs | (abs-mu)/sd] per generic subject, baseline = that subject's CALM rows.
    Identical to the deviation the generic model was trained with."""
    out = np.zeros((len(X), X.shape[1] * 2))
    for s in set(subj.tolist()):
        m = subj == s
        calm = m & (y == 0)
        base = X[calm] if calm.sum() >= 3 else X[m]
        mu, sd = base.mean(0), base.std(0) + 1e-8
        out[m] = np.hstack([X[m], (X[m] - mu) / sd])
    return out


def _personal_baseline() -> tuple[np.ndarray, np.ndarray] | None:
    """The mu/sd the live classifier uses, so personal rows deviate the same way."""
    if not _BASELINE_FILE.exists():
        return None
    try:
        d = np.load(_BASELINE_FILE)
        mu, sd = d["mu"], d["sd"]
        if mu.shape == (_N_FEATURES,) and sd.shape == (_N_FEATURES,):
            return mu, sd
    except Exception:
        pass
    return None


def _personal_rows() -> tuple[np.ndarray, np.ndarray]:
    """User's labelled windows that have a usable 7-feature state vector and a
    valid 3-state label. Returns (X raw [n,7], y [n])."""
    X, y = [], []
    for row in load_feedback():
        feats = row.get("state_features")
        code = row.get("state_code")
        if not feats or len(feats) != _N_FEATURES:
            continue
        if code not in (0, 1, 2):
            continue
        X.append([float(v) for v in feats])
        y.append(int(code))
    if not X:
        return np.zeros((0, _N_FEATURES)), np.zeros((0,), dtype=int)
    return np.array(X, dtype=float), np.array(y, dtype=int)


def _counts(y: np.ndarray) -> dict[str, int]:
    return {CLASSES[c]: int((y == c).sum()) for c in (0, 1, 2)}


def maybe_retrain(min_per_state: int = PERSONAL_MIN_PER_STATE) -> dict:
    """Train a personal model if there are >= min_per_state labels in EVERY state
    and it beats the generic model on a held-out slice of the user's own labels.

    Returns a dict the engine logs/acts on:
      {"trained": bool, "reason": str, "counts": {...},
       "self_check": {"personal": acc, "generic": acc} | None, "path": str | None}
    """
    if joblib is None:
        return {"trained": False, "reason": "sklearn/joblib unavailable",
                "counts": {}, "self_check": None, "path": None}

    Xp, yp = _personal_rows()
    counts = _counts(yp)
    if any(counts[CLASSES[c]] < min_per_state for c in (0, 1, 2)):
        return {"trained": False, "reason": "not enough labels per state",
                "counts": counts, "self_check": None, "path": None}

    gpath = generic_trainset_path()
    if not gpath.exists():
        return {"trained": False, "reason": f"generic trainset missing: {gpath.name}",
                "counts": counts, "self_check": None, "path": None}

    pers_base = _personal_baseline()
    if pers_base is None:
        return {"trained": False, "reason": "personal baseline not ready",
                "counts": counts, "self_check": None, "path": None}
    mu, sd = pers_base

    g = np.load(gpath)
    Xg, yg, subjg = g["X"], g["y"], g["subj"]
    Xg_dev = _generic_deviate(Xg, subjg, yg)
    # Personal rows deviated with the SAME baseline the live classifier serves with.
    Xp_dev = np.hstack([Xp, (Xp - mu) / sd])

    # Self-check: hold out one personal row per state (round-robin), train on the
    # rest + generic, compare personal vs generic-only on the held-out personal rows.
    test_mask = np.zeros(len(yp), dtype=bool)
    for c in (0, 1, 2):
        idx = np.where(yp == c)[0]
        if len(idx) >= 2:  # keep at least one for training
            n_hold = max(1, len(idx) // 4)  # ~25% held out, >=1
            test_mask[idx[:n_hold]] = True
    tr = ~test_mask

    self_check = None
    if test_mask.sum() >= 2 and len(set(yp[tr].tolist())) == 3:
        Xtr = np.vstack([Xg_dev, Xp_dev[tr]])
        ytr = np.concatenate([yg, yp[tr]])
        c_pers = _rf().fit(Xtr, ytr)
        acc_pers = balanced_accuracy_score(yp[test_mask], c_pers.predict(Xp_dev[test_mask])) * 100
        c_gen = _rf().fit(Xg_dev, yg)
        acc_gen = balanced_accuracy_score(yp[test_mask], c_gen.predict(Xp_dev[test_mask])) * 100
        self_check = {"personal": round(acc_pers, 1), "generic": round(acc_gen, 1)}
        if acc_pers < acc_gen:
            return {"trained": False, "reason": "personal did not beat generic",
                    "counts": counts, "self_check": self_check, "path": None}

    # Passed (or too few to check): train the FINAL personal model on everything.
    Xall = np.vstack([Xg_dev, Xp_dev])
    yall = np.concatenate([yg, yp])
    final = _rf().fit(Xall, yall)
    bundle = {
        "model": final,
        "feature_names": ["rmssd", "mean_hr", "sdnn", "mean_ibi", "pnn50",
                          "acc_rms", "acc_std"],
        "uses_personal_deviation": True,
        "classes": CLASSES,
        "trained_on": f"generic EMOGNITION + {len(yp)} personal labels",
        "personal_counts": counts,
        "self_check": self_check,
        "window_s": 20,
    }
    out = personal_model_path()
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out)
    return {"trained": True, "reason": "personal model trained",
            "counts": counts, "self_check": self_check, "path": str(out)}
