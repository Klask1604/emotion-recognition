#!/usr/bin/env python3
"""
GalaxyPPG validation report.

Reads the JSONL trajectory written by tools/replay_galaxyppg.py and computes
quantitative agreement between our wrist-only pipeline and the Polar H10 ECG
ground truth, plus per-session arousal behaviour against expected stimuli
intensities. Outputs a markdown report under eval_results/.

Metrics:
  - Coverage: fraction of session ticks where the GalaxyWatch HR_CONTINUOUS
    tracker produced enough IBI for our pipeline to emit a decision.
  - RMSSD agreement: Pearson r and MAE between our_rmssd and the Polar H10
    RMSSD computed over the same 30 s window (ground truth).
  - Per-session arousal: mean arousal_10 per session vs the expected value
    encoded in SESSION_EXPECTED_AROUSAL (subjective IAPS-style estimate).
  - Stress detection: classify each session-second as stress (tsst-speech,
    ssst-sing) vs rest (baseline, rest-*, meditation-*) using arousal_10
    above a threshold; report sensitivity, specificity, balanced accuracy.

Run:
  python tools/replay_galaxyppg.py            # produces eval_results/galaxyppg_replay.jsonl
  python scripts/validate_galaxyppg.py        # consumes that, writes the report
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.replay_galaxyppg import SESSION_EXPECTED_AROUSAL  # noqa: E402

STRESS_SESSIONS = {"tsst-speech", "ssst-sing"}
REST_SESSIONS = {
    "baseline", "rest-1", "rest-2", "rest-3", "rest-4", "rest-5",
    "meditation-1", "meditation-2",
}

DEFAULT_AROUSAL_STRESS_THRESHOLD = 6  # 1..10 → ≥6 means "Moderate or higher"


def pearson_r(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return float("nan")
    mx = mean(xs); my = mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx2 = sum((x - mx) ** 2 for x in xs)
    dy2 = sum((y - my) ** 2 for y in ys)
    denom = math.sqrt(dx2 * dy2)
    return num / denom if denom > 0 else float("nan")


def mae(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return float("nan")
    return sum(abs(x - y) for x, y in zip(xs, ys)) / len(xs)


def fmt(v: float, dec: int = 2) -> str:
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "—"
    return f"{v:.{dec}f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="eval_results/galaxyppg_replay.jsonl")
    parser.add_argument("--output", default=None,
                        help="Default: eval_results/galaxyppg_validation_<date>.md")
    parser.add_argument("--threshold", type=int, default=DEFAULT_AROUSAL_STRESS_THRESHOLD,
                        help="arousal_10 threshold above which the sample is "
                             "classified as 'stress' (default: 6).")
    args = parser.parse_args()

    in_path = ROOT / args.input
    if not in_path.exists():
        raise SystemExit(f"missing {in_path} — run tools/replay_galaxyppg.py first")
    if args.output:
        out_path = ROOT / args.output
    else:
        out_path = ROOT / f"eval_results/galaxyppg_validation_{datetime.now().strftime('%Y%m%d_%H%M')}.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    samples_by_subj: dict[str, list[dict]] = defaultdict(list)
    for line in in_path.open("r", encoding="utf-8"):
        s = json.loads(line)
        samples_by_subj[s["subject"]].append(s)
    subjects = sorted(samples_by_subj)

    # ── 1. Coverage per subject ──────────────────────────────────────────────
    coverage_rows: list[tuple[str, int, int, float]] = []
    for subj in subjects:
        total = len(samples_by_subj[subj])
        with_dec = sum(1 for s in samples_by_subj[subj] if s["arousal_10"] is not None)
        pct = 100.0 * with_dec / total if total else 0.0
        coverage_rows.append((subj, total, with_dec, pct))

    # ── 2. RMSSD agreement per subject (where both ours + polar exist) ───────
    agreement_rows: list[tuple[str, int, float, float, float, float]] = []
    for subj in subjects:
        pairs = [
            (s["our_rmssd"], s["polar_rmssd"])
            for s in samples_by_subj[subj]
            if s["our_rmssd"] and s["polar_rmssd"] and s["our_rmssd"] > 0 and s["polar_rmssd"] > 0
        ]
        n = len(pairs)
        if n < 5:
            agreement_rows.append((subj, n, float("nan"), float("nan"),
                                   float("nan"), float("nan")))
            continue
        our = [p[0] for p in pairs]
        pol = [p[1] for p in pairs]
        r = pearson_r(our, pol)
        m = mae(our, pol)
        bias = mean(our) - mean(pol)
        agreement_rows.append((subj, n, r, m, bias, mean(our) / mean(pol)))

    # ── 3. Per-session arousal across all subjects ───────────────────────────
    session_arousal: dict[str, list[int]] = defaultdict(list)
    session_our_rmssd: dict[str, list[float]] = defaultdict(list)
    session_polar_rmssd: dict[str, list[float]] = defaultdict(list)
    session_sq: dict[str, list[float]] = defaultdict(list)
    for subj_samples in samples_by_subj.values():
        for s in subj_samples:
            sess = s["session"]
            if not sess:
                continue
            if s["arousal_10"] is not None:
                session_arousal[sess].append(s["arousal_10"])
                session_our_rmssd[sess].append(s["our_rmssd"] or 0)
                session_sq[sess].append(s["signal_quality"])
            if s["polar_rmssd"]:
                session_polar_rmssd[sess].append(s["polar_rmssd"])

    # Order sessions by their appearance in the protocol (canonical order)
    session_order = list(SESSION_EXPECTED_AROUSAL.keys())
    extra = sorted(set(session_arousal) - set(session_order))
    session_order = [s for s in session_order if s in session_arousal] + extra

    # ── 4. Stress detection (per-tick label) ─────────────────────────────────
    tp = fn = fp = tn = 0
    for subj_samples in samples_by_subj.values():
        for s in subj_samples:
            sess = s["session"]
            a = s["arousal_10"]
            if a is None:
                continue
            if sess in STRESS_SESSIONS:
                if a >= args.threshold:
                    tp += 1
                else:
                    fn += 1
            elif sess in REST_SESSIONS:
                if a >= args.threshold:
                    fp += 1
                else:
                    tn += 1
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    bal_acc = (sens + spec) / 2 if not math.isnan(sens) and not math.isnan(spec) else float("nan")

    # ── Render markdown ──────────────────────────────────────────────────────
    md: list[str] = []
    md.append(f"# GalaxyPPG validation report")
    md.append("")
    md.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    md.append(f"Input: `{args.input}` ({sum(len(v) for v in samples_by_subj.values())} ticks, "
              f"{len(subjects)} subjects)")
    md.append("")
    md.append("## 1. Pipeline decision coverage per subject")
    md.append("")
    md.append("How often the GalaxyWatch HR_CONTINUOUS tracker produced enough IBI "
              "for the pipeline to emit a decision. Low coverage = tracker silent "
              "during many sessions (motion, weak PPG, sensor reset).")
    md.append("")
    md.append("| Subject | Session ticks | Ticks with decision | Coverage % |")
    md.append("|---|---:|---:|---:|")
    for subj, total, with_dec, pct in coverage_rows:
        md.append(f"| {subj} | {total} | {with_dec} | {pct:.1f}% |")
    avg_cov = mean(r[3] for r in coverage_rows)
    md.append(f"| **mean** | — | — | **{avg_cov:.1f}%** |")
    md.append("")

    md.append("## 2. RMSSD agreement vs Polar H10 (ECG ground truth)")
    md.append("")
    md.append("Per subject, Pearson correlation and Mean Absolute Error between our "
              "computed RMSSD and the Polar H10 RMSSD over the same 30 s window. "
              "Subjects with <5 paired samples are reported as —.")
    md.append("")
    md.append("| Subject | N pairs | Pearson r | MAE (ms) | Bias (ms) | Ratio ours/polar |")
    md.append("|---|---:|---:|---:|---:|---:|")
    for subj, n, r, m, b, ratio in agreement_rows:
        md.append(f"| {subj} | {n} | {fmt(r, 3)} | {fmt(m, 1)} | {fmt(b, 1)} | {fmt(ratio, 2)} |")
    valid = [a for a in agreement_rows if not math.isnan(a[2])]
    if valid:
        md.append("")
        md.append(f"- **Mean Pearson r**: {mean(a[2] for a in valid):.3f}")
        md.append(f"- **Mean MAE**: {mean(a[3] for a in valid):.1f} ms")
        md.append(f"- **Mean bias (ours − polar)**: {mean(a[4] for a in valid):+.1f} ms")
        md.append(f"- **Mean ratio**: {mean(a[5] for a in valid):.2f}× (ours vs polar; >1 means we overestimate)")
    md.append("")

    md.append("## 3. Mean arousal_10 per session (across all subjects)")
    md.append("")
    md.append("Compares the pipeline's verdict against an a-priori expected arousal "
              "for each session type. Expected values are subjective placements on "
              "the 1..10 scale (lower=relaxed, higher=stressed/active).")
    md.append("")
    md.append("| Session | N ticks | Mean arousal | Expected | Δ | Mean our RMSSD | Mean Polar RMSSD | Mean SQ |")
    md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for sess in session_order:
        arousal_vals = session_arousal.get(sess, [])
        n = len(arousal_vals)
        if n == 0:
            continue
        exp = SESSION_EXPECTED_AROUSAL.get(sess, float("nan"))
        actual = mean(arousal_vals)
        delta = actual - exp if not math.isnan(exp) else float("nan")
        our_rmssd_vals = session_our_rmssd.get(sess, [])
        polar_rmssd_vals = session_polar_rmssd.get(sess, [])
        sq_vals = session_sq.get(sess, [])
        md.append(f"| {sess} | {n} | {actual:.2f} | "
                  f"{fmt(exp, 1)} | {fmt(delta, 2)} | "
                  f"{mean(our_rmssd_vals):.1f} | "
                  f"{mean(polar_rmssd_vals) if polar_rmssd_vals else 0:.1f} | "
                  f"{mean(sq_vals):.2f} |")
    md.append("")

    md.append(f"## 4. Stress detection accuracy (threshold arousal_10 ≥ {args.threshold})")
    md.append("")
    md.append(f"Stress sessions: {sorted(STRESS_SESSIONS)}")
    md.append(f"Rest sessions: {sorted(REST_SESSIONS)}")
    md.append("")
    md.append("| | predicted stress | predicted rest |")
    md.append("|---|---:|---:|")
    md.append(f"| **actually stress** | TP = {tp} | FN = {fn} |")
    md.append(f"| **actually rest**   | FP = {fp} | TN = {tn} |")
    md.append("")
    md.append(f"- Sensitivity (TP / TP+FN): **{fmt(sens, 3)}**")
    md.append(f"- Specificity (TN / TN+FP): **{fmt(spec, 3)}**")
    md.append(f"- Balanced accuracy: **{fmt(bal_acc, 3)}**")
    md.append("")

    md.append("## 5. Notes / limitations")
    md.append("")
    md.append("- The GalaxyWatch HR_CONTINUOUS tracker on the dataset device does "
              "not deliver IBI continuously: rest periods and motion-heavy sessions "
              "often contain zero usable bursts. This is reflected in the coverage "
              "table above. It mirrors what we observe in production wear-time.")
    md.append("- Pre-baseline ticks are reported with the population Kubios zone "
              "(`decision_fidelity = preliminary`, confidence capped at 0.5).")
    md.append("- Polar H10 timestamps in the dataset arrive offset by +9 h "
              "(timezone UTC+9 stored as raw epoch ms by the Polar Sensor Logger). "
              "The replay subtracts a whole-hour rounded offset automatically.")
    md.append("- Expected arousal values are subjective placeholders, not validated "
              "norms; they exist to surface large mismatches, not to grade absolute "
              "accuracy. Compare relative ordering across sessions for trends.")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
