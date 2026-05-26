#!/usr/bin/env python3
"""
WESAD-vs-deterministic comparison metrics, computed from live InfluxDB data.

Compares the parallel WESAD RandomForest (biofizic_legacy_wesad.p_stress) against
the production deterministic verdict (biofizic_state: arousal_10 + alert) over a
recent window, and reports:

  - Cohen's kappa            agreement between the two binary stress calls
  - WESAD false-positive rate fraction of REST epochs (ours says calm) where
                              WESAD nonetheless says stress  -> the domain-shift
                              over-firing we expect from a foreign-dataset model
  - mean p_stress at rest    how "stressed" WESAD thinks you are while calm
  - Spearman correlation      p_stress vs arousal_10

Binary calls: ours = (alert OR arousal_10 >= threshold); WESAD = p_stress >= 0.5.

Usage:
    python scripts/wesad_comparison_report.py --hours 24
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401
from biofizic.engine.arousal_mapper import cohen_kappa


def query_sql(url: str, database: str, sql: str) -> list[dict]:
    payload = json.dumps({"db": database, "q": sql, "format": "json"}).encode()
    req = urllib.request.Request(
        f"{url.rstrip('/')}/api/v3/query_sql",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.load(resp)


def _epoch(ts: str) -> float:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _nearest(sorted_ts: list[float], values: list[float], t: float, tol: float) -> float | None:
    """Nearest value whose timestamp is within tol seconds of t (linear scan)."""
    best, best_dt = None, tol
    for ts, v in zip(sorted_ts, values):
        d = abs(ts - t)
        if d <= best_dt:
            best, best_dt = v, d
    return best


def _spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    try:
        from scipy.stats import spearmanr

        rho = spearmanr(x, y).correlation
        return float(rho) if rho == rho else float("nan")
    except Exception:
        return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8181")
    ap.add_argument("--database", default="biofizic")
    ap.add_argument("--hours", type=float, default=24.0, help="lookback window")
    ap.add_argument("--arousal-threshold", type=int, default=7,
                    help="our arousal_10 >= this counts as 'stress'")
    ap.add_argument("--align-tolerance-s", type=float, default=20.0)
    ap.add_argument("--output", default=str(ROOT / "eval_results" / "wesad_comparison.json"))
    args = ap.parse_args()

    horizon = f"now() - interval '{args.hours} hours'"
    state = query_sql(
        args.url, args.database,
        f"SELECT time, arousal_10, alert FROM biofizic_state "
        f"WHERE time >= {horizon} AND arousal_10 IS NOT NULL ORDER BY time",
    )
    wesad = query_sql(
        args.url, args.database,
        f"SELECT time, p_stress FROM biofizic_legacy_wesad "
        f"WHERE time >= {horizon} AND p_stress IS NOT NULL ORDER BY time",
    )

    if len(state) < 3 or len(wesad) < 3:
        print(f"Not enough paired data (state={len(state)}, wesad={len(wesad)}). "
              f"Enable ENABLE_WESAD and run a session first.")
        sys.exit(2)

    w_ts = [_epoch(r["time"]) for r in wesad]
    w_p = [float(r["p_stress"]) for r in wesad]

    ours_labels, wes_labels, arousals, pstresses = [], [], [], []
    for r in state:
        t = _epoch(r["time"])
        p = _nearest(w_ts, w_p, t, args.align_tolerance_s)
        if p is None:
            continue
        a10 = float(r["arousal_10"])
        alert = float(r.get("alert") or 0.0) >= 0.5
        ours = "stress" if (alert or a10 >= args.arousal_threshold) else "calm"
        wes = "stress" if p >= 0.5 else "calm"
        ours_labels.append(ours)
        wes_labels.append(wes)
        arousals.append(a10)
        pstresses.append(p)

    n = len(ours_labels)
    if n < 3:
        print(f"Only {n} aligned pairs within {args.align_tolerance_s}s — too few.")
        sys.exit(2)

    agree = sum(a == b for a, b in zip(ours_labels, wes_labels)) / n
    kappa = cohen_kappa(ours_labels, wes_labels)
    rest = [w for o, w in zip(ours_labels, wes_labels) if o == "calm"]
    fp_rate = (sum(w == "stress" for w in rest) / len(rest)) if rest else float("nan")
    rest_p = [p for o, p in zip(ours_labels, pstresses) if o == "calm"]
    mean_p_rest = (sum(rest_p) / len(rest_p)) if rest_p else float("nan")
    rho = _spearman(pstresses, arousals)

    report = {
        "window_hours": args.hours,
        "n_pairs": n,
        "arousal_threshold": args.arousal_threshold,
        "agreement": round(agree, 3),
        "cohen_kappa": round(kappa, 3),
        "wesad_false_positive_rate_at_rest": round(fp_rate, 3) if fp_rate == fp_rate else None,
        "mean_p_stress_at_rest": round(mean_p_rest, 3) if mean_p_rest == mean_p_rest else None,
        "spearman_p_stress_vs_arousal": round(rho, 3) if rho == rho else None,
    }

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print("=== WESAD vs deterministic (live) ===")
    for k, v in report.items():
        print(f"  {k:38s} {v}")
    print(f"\nSaved {out}")
    print(
        "\nInterpretation: low kappa + high false-positive-rate-at-rest means the "
        "WESAD model (chest ECG, foreign subjects) over-fires on your wrist data "
        "-- the domain-shift evidence for preferring the personal deterministic model."
    )


if __name__ == "__main__":
    main()
