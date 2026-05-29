#!/usr/bin/env python3
"""
One-shot copy of InfluxDB tables from the SERVER to a LOCAL InfluxDB, so
dashboards can be edited locally against real data without the flaky external
FlightSQL/gRPC path. HTTP query out of the server, line-protocol write into
local. Idempotent-ish: re-running appends, so wipe local first if you want a
clean copy.

Usage:
    ./venv/Scripts/python.exe tools/copy_influx_server_to_local.py
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from datetime import datetime, timezone

SERVER = "http://paxbespoke.automateflow.ro:8181"
LOCAL = "http://localhost:8181"
DB = "biofizic"

# String columns per table (everything else is treated as a float field).
STRING_COLS = {
    "biofizic_live": {"motion_state", "dominant_channel", "decision_fidelity"},
    "biofizic_state": {"motion_state", "dominant_channel", "decision_fidelity",
                       "emotion", "emotion_baseline", "engine", "why",
                       "window_used", "data_quality"},
    "biofizic_state_live": {"motion_state", "dominant_channel", "decision_fidelity",
                            "engine", "window_used", "data_quality"},
}

TABLES = [
    "biofizic_live",
    "biofizic_legacy_valence",
    "biofizic_legacy_resp",
    "biofizic_legacy_ppg",
    "biofizic_state",
    "biofizic_state_live",
    "biofizic_state_windows",
]


def server_query(sql: str) -> list[dict]:
    url = SERVER + "/api/v3/query_sql?" + urllib.parse.urlencode(
        {"q": sql, "db": DB, "format": "json"}
    )
    return json.load(urllib.request.urlopen(url, timeout=120))


def to_line_protocol(measurement: str, rows: list[dict], string_cols: set[str]) -> str:
    lines = []
    for r in rows:
        t = r.get("time")
        if not t:
            continue
        # Parse ISO time -> epoch ns.
        ts = t.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(ts)
        except ValueError:
            # Truncate fractional seconds beyond microseconds if present.
            dt = datetime.fromisoformat(ts[:26])
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        epoch_ns = int(dt.timestamp() * 1_000_000_000)

        tags = []
        fields = []
        for k, v in r.items():
            if k == "time" or v is None:
                continue
            if k in string_cols:
                # escape spaces/commas in tag values
                sv = str(v).replace(" ", "\\ ").replace(",", "\\,")
                tags.append(f"{k}={sv}")
            else:
                try:
                    fields.append(f"{k}={float(v)}")
                except (TypeError, ValueError):
                    sv = str(v).replace('"', '\\"')
                    fields.append(f'{k}="{sv}"')
        if not fields:
            continue
        tagstr = ("," + ",".join(tags)) if tags else ""
        lines.append(f"{measurement}{tagstr} {','.join(fields)} {epoch_ns}")
    return "\n".join(lines)


def local_write(lines: str) -> None:
    if not lines:
        return
    url = LOCAL + "/api/v3/write_lp?" + urllib.parse.urlencode(
        {"db": DB, "precision": "nanosecond"}
    )
    req = urllib.request.Request(url, data=lines.encode(), method="POST")
    urllib.request.urlopen(req, timeout=120)


def main() -> None:
    for table in TABLES:
        try:
            rows = server_query(f"SELECT * FROM {table}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {table}: SERVER read failed: {exc}")
            continue
        if not rows:
            print(f"  {table}: empty on server, skipped")
            continue
        scols = STRING_COLS.get(table, set())
        # Write in chunks so a single payload is not huge.
        n = 0
        CHUNK = 5000
        for i in range(0, len(rows), CHUNK):
            lp = to_line_protocol(table, rows[i:i + CHUNK], scols)
            try:
                local_write(lp)
                n += lp.count("\n") + 1 if lp else 0
            except Exception as exc:  # noqa: BLE001
                print(f"  {table}: LOCAL write failed at chunk {i}: {exc}")
                break
        print(f"  {table}: copied ~{n} rows")
    print("done.")


if __name__ == "__main__":
    sys.exit(main())
