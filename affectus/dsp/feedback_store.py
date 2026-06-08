"""Append-only store of user emotion-feedback labels.

When the user taps the feedback button on the watch and picks a Russell quadrant,
the server pairs that label with the 33-feature PPG vector from the window the
user was feeling it in, plus what each valence model predicted at that moment.
Each pair is one JSONL line in data/feedback_labels.jsonl, the watch-native
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

# The three states the live models use, matching their class codes
# (CALM=0, DISCONFORT=1, PLACUT=2 in stari_3.joblib / morfologie_3.joblib).
STATES = ("Calm", "Disconfort", "Placut")
STATE_CODE = {"Calm": 0, "Disconfort": 1, "Placut": 2}

# Old 4-quadrant taps from a watch that has not been rebuilt yet. Map them onto the
# three states so feedback keeps working through the transition. The two negative
# quadrants both collapse to Disconfort (valence is the axis the model splits);
# Bucuros and Calm are both pleasant, so Bucuros -> Placut, Calm -> Calm.
_LEGACY_QUADRANT_TO_STATE = {
    "Calm": "Calm",
    "Bucuros": "Placut",
    "Trist": "Disconfort",
    "Stresat": "Disconfort",
}


def normalize_state(label: str) -> str | None:
    """Accept a new 3-state label or an old 4-quadrant one. Returns the canonical
    state name, or None if the label is unknown."""
    if label in STATE_CODE:
        return label
    return _LEGACY_QUADRANT_TO_STATE.get(label)


# Back-compat aliases so existing imports keep working during the transition.
QUADRANTS = STATES
QUADRANT_CODE = STATE_CODE


def default_path() -> Path:
    # Use the shared data_dir() so it lands in the mounted /data volume inside
    # Docker (persists across restarts), NOT /app/data which is baked + ephemeral.
    from affectus.config import data_dir
    return data_dir() / "feedback_labels.jsonl"


def append_feedback(
    quadrant: str,
    features: list[float] | None,
    *,
    arousal_z: float | None = None,
    features_age_s: float | None = None,
    preds: dict | None = None,
    state_features: list[float] | None = None,
    morpho_features: list[float] | None = None,
    path: Path | None = None,
) -> dict:
    """Append one labelled row. Returns the row dict (also used for re-publish).

    quadrant       : the user's chosen label (new 3-state, or old 4-quadrant which
                     gets mapped). Stored canonically as a state in `state`/`state_code`.
    features       : the legacy 33-feature valence vector, if available (kept for the
                     old dashboards; the retrain loop does not use it).
    arousal_z      : the live arousal z at label time.
    features_age_s : how old the feature window is vs the tap.
    preds          : {"wesad": p_pos, ...} legacy model predictions at label time.
    state_features : the HRV state model's live feature vector (9 numbers) at label
                     time. THIS is what the personal retrain uses.
    morpho_features: the morphology model's live feature vector (8 numbers), if fresh.
    """
    p = path or default_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    preds = preds or {}
    state = normalize_state(quadrant)
    row = {
        "ts": int(time.time() * 1000),
        "quadrant": quadrant,
        "quadrant_code": QUADRANT_CODE.get(quadrant, -1),
        "state": state,
        "state_code": STATE_CODE.get(state, -1) if state else -1,
        "arousal_z": round(arousal_z, 4) if arousal_z is not None else None,
        "features_age_s": round(features_age_s, 2) if features_age_s is not None else None,
        "n_features": len(features) if features else 0,
        "features": [round(float(x), 6) for x in features] if features else None,
        "state_features": [round(float(x), 6) for x in state_features] if state_features else None,
        "morpho_features": [round(float(x), 6) for x in morpho_features] if morpho_features else None,
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
