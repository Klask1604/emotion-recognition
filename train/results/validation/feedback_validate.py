#!/usr/bin/env python3
"""Validate the 3 valence models against the user's OWN feedback labels.

Phase A deliverable: every time the user taps the feedback button on the watch,
the server stored [the user's Russell quadrant] + [what WESAD/EEVR/CASE predicted
at that moment] in data/feedback_labels.jsonl. This script reads those rows and
reports, per model, how often its prediction matched the user's truth — an HONEST
accuracy on the user's own watch data, not a borrowed dataset.

Each model outputs p_positive (probability of positive VALENCE). The user's label
encodes both axes, but the models only claim valence, so we score on the VALENCE
axis: positive quadrants (Bucuros, Calm) vs negative (Trist, Stresat).
  model predicts positive  if p_positive >= 0.5
  user label is positive   if quadrant in (Bucuros, Calm)

Reports: n labels, per-model balanced accuracy + confusion (pos/neg), and how
often the user's own arousal_z agreed with the quadrant's arousal axis (a sanity
check that arousal — the axis that works — tracks the user's report).

Run: ./venv/Scripts/python.exe train/feedback_validate.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from affectus.dsp.feedback_store import load_feedback  # noqa: E402

POSITIVE_VALENCE = {"Bucuros", "Calm"}      # pleasant quadrants
HIGH_AROUSAL = {"Bucuros", "Stresat"}       # activated quadrants


def _valence_true(quadrant: str) -> int | None:
    if quadrant in POSITIVE_VALENCE:
        return 1
    if quadrant in ("Trist", "Stresat"):
        return 0
    return None


def _balanced_acc(pairs: list[tuple[int, int]]) -> tuple[float, dict]:
    """pairs = [(true, pred)]. Returns balanced accuracy + confusion counts."""
    conf = Counter()
    for t, p in pairs:
        conf[(t, p)] += 1
    out = {}
    accs = []
    for cls in (0, 1):
        tp = conf[(cls, cls)]
        n = conf[(cls, 0)] + conf[(cls, 1)]
        out[cls] = (tp, n)
        if n:
            accs.append(tp / n)
    bacc = sum(accs) / len(accs) if accs else float("nan")
    return bacc, out


def main() -> None:
    rows = load_feedback()
    if not rows:
        print("No feedback labels yet. Tap the feedback button on the watch first.")
        print(f"(expected file: {ROOT / 'data' / 'feedback_labels.jsonl'})")
        return

    print(f"=== Feedback validation — {len(rows)} labels ===")
    qcount = Counter(r["quadrant"] for r in rows)
    print("labels per quadrant:", dict(qcount))
    n_with_feat = sum(1 for r in rows if r.get("n_features"))
    print(f"labels with usable PPG features: {n_with_feat}/{len(rows)}\n")

    # Per-model valence accuracy vs the user's own truth.
    print("VALENCE axis — model prediction vs YOUR label (positive=Bucuros/Calm):")
    for model, key in (("WESAD", "wesad_p_positive"),
                       ("EEVR", "eevr_p_positive"),
                       ("CASE", "case_p_positive")):
        pairs = []
        for r in rows:
            yt = _valence_true(r["quadrant"])
            p = r.get(key)
            if yt is None or not isinstance(p, (int, float)):
                continue
            pairs.append((yt, 1 if p >= 0.5 else 0))
        if not pairs:
            print(f"  {model:6s}: no comparable labels yet")
            continue
        bacc, conf = _balanced_acc(pairs)
        pos_tp, pos_n = conf.get(1, (0, 0))
        neg_tp, neg_n = conf.get(0, (0, 0))
        print(f"  {model:6s}: balanced acc {bacc*100:5.1f}%  "
              f"(positive {pos_tp}/{pos_n}, negative {neg_tp}/{neg_n}, n={len(pairs)})")

    # Sanity: does the user's arousal_z agree with the quadrant's arousal axis?
    aro_pairs = []
    for r in rows:
        az = r.get("arousal_z")
        if not isinstance(az, (int, float)):
            continue
        true_high = 1 if r["quadrant"] in HIGH_AROUSAL else 0
        aro_pairs.append((true_high, 1 if az >= 0 else 0))
    if aro_pairs:
        bacc, _ = _balanced_acc(aro_pairs)
        print(f"\nAROUSAL sanity — measured arousal_z vs quadrant arousal axis: "
              f"{bacc*100:.1f}% (n={len(aro_pairs)})")
        print("  (this is the axis that WORKS — high if the watch's arousal tracks "
              "what you reported)")

    print(f"\n{n_with_feat} feature vectors banked in data/feedback_labels.jsonl. "
          f"At ~20-30 with varied quadrants, Phase B can test a PERSONAL model.")


if __name__ == "__main__":
    main()
