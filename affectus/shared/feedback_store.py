"""Append-only store of user emotion-feedback labels.

When the user taps the feedback button on the watch and picks a Russell quadrant,
the server pairs that label with the 33-feature PPG vector from the window the
user was feeling it in, plus what each valence model predicted at that moment.
Each pair is one JSONL line in data/feedback_labels.jsonl — the watch-native
labelled dataset that the whole cross-subject research lacked.

Phase A consumes this for validation (model vs the user's truth). Phase B may
later train a PERSONAL model on it, IF the signal turns out to exist for this
subject. We never throw a row away (the user may re-label); analysis filters by ts.

One function, append-only, so a crash mid-write loses at most the current line.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

# Russell quadrant -> (arousal sign, valence sign) target, the ground truth the
# label encodes on BOTH axes. Codes match the watch/dashboard value mappings.
QUADRANTS = ("Bucuros", "Calm", "Trist", "Stresat")
QUADRANT_CODE = {"Bucuros": 3, "Calm": 1, "Trist": 2, "Stresat": 4}


def default_path() -> Path:
    # Use the shared data_dir() so it lands in the mounted /data volume inside
    # Docker (persists across restarts) — NOT /app/data which is baked + ephemeral.
    from affectus.config import data_dir
    return data_dir() / "feedback_labels.jsonl"


def append_feedback(
    quadrant: str,
    features: list[float] | None,
    *,
    arousal_z: float | None = None,
    features_age_s: float | None = None,
    preds: dict | None = None,
    path: Path | None = None,
) -> dict:
    """Append one labelled row. Returns the row dict (also used for re-publish).

    quadrant      : one of QUADRANTS (the user's chosen state).
    features      : the 33-feature PPG vector from the labelled window, or None if
                    no fresh window was available (still stored — a label with no
                    usable signal is itself informative).
    arousal_z     : the live arousal z at label time (so the row carries both axes).
    features_age_s: how old the feature window is vs the tap (freshness for filtering).
    preds         : {"wesad": p_pos, "eevr": p_pos, "case": p_pos} model predictions
                    at label time, for the model-vs-truth validation.
    """
    p = path or default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    preds = preds or {}
    row = {
        "ts": int(time.time() * 1000),
        "quadrant": quadrant,
        "quadrant_code": QUADRANT_CODE.get(quadrant, -1),
        "arousal_z": round(arousal_z, 4) if arousal_z is not None else None,
        "features_age_s": round(features_age_s, 2) if features_age_s is not None else None,
        "n_features": len(features) if features else 0,
        "features": [round(float(x), 6) for x in features] if features else None,
        "wesad_p_positive": _r(preds.get("wesad")),
        "eevr_p_positive": _r(preds.get("eevr")),
        "case_p_positive": _r(preds.get("case")),
    }
    with open(p, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")
    return row


def _r(v):
    return round(float(v), 4) if isinstance(v, (int, float)) else None


def load_feedback(path: Path | None = None) -> list[dict]:
    """Read all labelled rows (skips blank/corrupt lines). Empty list if none yet."""
    p = path or default_path()
    if not p.exists():
        return []
    rows = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows
