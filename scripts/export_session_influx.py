#!/usr/bin/env python3
"""Export InfluxDB session data (fusion + epoch) to CSV for offline review."""

from __future__ import annotations

import argparse
import csv
import json
import urllib.request
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

LOCAL_TZ = ZoneInfo("Europe/Bucharest")


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


def to_local(ts: str) -> str:
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(LOCAL_TZ).strftime("%Y-%m-%d %H:%M:%S")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export Biofizic session from InfluxDB")
    parser.add_argument("--url", default="http://localhost:8181")
    parser.add_argument("--database", default="biofizic")
    parser.add_argument("--from-utc", required=True, help="ISO UTC start, e.g. 2026-05-23T09:00:00Z")
    parser.add_argument("--to-utc", required=True, help="ISO UTC end, e.g. 2026-05-23T12:14:00Z")
    parser.add_argument("-o", "--output", required=True)
    args = parser.parse_args()

    sql = f"""
    SELECT
      time,
      hr,
      rmssd,
      rmssd_ppg,
      arousal_fused,
      arousal_10,
      confidence_fused,
      confidence_v2,
      confidence_v3,
      w_v2,
      w_v3,
      systems_agree,
      strong_agreement,
      emotion_v2,
      emotion_v3,
      arousal_label,
      motion,
      acc_rms,
      z_pulse,
      AVG(systems_agree) OVER (
        ORDER BY time ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
      ) * 100 AS agree_pct
    FROM biofizic_combined
    WHERE time >= '{args.from_utc}' AND time <= '{args.to_utc}'
      AND hr > 0
    ORDER BY time
    """
    rows = query_sql(args.url, args.database, sql)
    if not rows:
        raise SystemExit("No rows returned for the selected interval.")

    fieldnames = [
        "time_utc",
        "time_local",
        "hr",
        "rmssd",
        "rmssd_ppg",
        "arousal_fused",
        "arousal_10",
        "confidence_fused",
        "confidence_v2",
        "confidence_v3",
        "w_v2",
        "w_v3",
        "systems_agree",
        "agree_pct",
        "strong_agreement",
        "emotion_v2",
        "emotion_v3",
        "arousal_label",
        "motion",
        "acc_rms",
        "z_pulse",
    ]

    with open(args.output, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "time_utc": row.get("time"),
                    "time_local": to_local(str(row.get("time", ""))),
                    **{k: row.get(k) for k in fieldnames if k not in ("time_utc", "time_local")},
                }
            )

    print(f"Exported {len(rows)} rows -> {args.output}")


if __name__ == "__main__":
    main()
