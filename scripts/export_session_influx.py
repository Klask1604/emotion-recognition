#!/usr/bin/env python3
"""Export InfluxDB session data (state + combined + sensors) to CSV."""

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
      arousal_10,
      valence_10,
      emotion,
      rmssd,
      stress_index,
      mean_hr,
      motion_class,
      activity_mode,
      confidence,
      z_pulse_amp,
      labels_agree
    FROM biofizic_state
    WHERE time >= '{args.from_utc}' AND time <= '{args.to_utc}'
    ORDER BY time
    """
    rows = query_sql(args.url, args.database, sql)
    if not rows:
        raise SystemExit("No rows returned for the selected interval.")

    fieldnames = [
        "time_utc",
        "time_local",
        "arousal_10",
        "valence_10",
        "emotion",
        "rmssd",
        "stress_index",
        "mean_hr",
        "motion_class",
        "activity_mode",
        "confidence",
        "z_pulse_amp",
        "labels_agree",
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
