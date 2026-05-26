#!/usr/bin/env python3
"""
Replay a recorded acquisition/batch session into an MQTT broker.

The recorder writes one JSON object per line of the form:
    {"recv_ts_ms": <int>, "payload": <acquisition/batch v2 dict>}

Replay preserves the original inter-batch spacing so the server sees the same
1 Hz cadence (or accelerated via --speed N). ts_publish and ts_anchor in the
payload are rewritten to be relative to wall-clock-now so InfluxDB rows do
not collide with the original session, while seq and motion stats are kept
intact.

Usage:
    python tools/replay_session.py session_2026-05-26.jsonl
    python tools/replay_session.py session.jsonl --broker localhost --speed 4

The script is meant to validate reproducibility: replay the same recording
twice and the compute-engine must produce the same decisions for each.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace", help="JSONL file produced by record_session.py")
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="Replay speed multiplier (2.0 means twice as fast)",
    )
    parser.add_argument(
        "--keep-timestamps",
        action="store_true",
        help="Do not rewrite ts_publish/ts_anchor (useful for byte-exact replay)",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    if not trace_path.exists():
        print(f"trace not found: {trace_path}", file=sys.stderr)
        sys.exit(1)

    client = mqtt.Client(
        client_id="biofizic_replayer",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.connect(args.broker, args.port, keepalive=60)
    client.loop_start()

    records = []
    with trace_path.open("r", encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"skipping malformed line {line_no}", file=sys.stderr)
    if not records:
        print("trace is empty", file=sys.stderr)
        sys.exit(1)

    base_recv = int(records[0]["recv_ts_ms"])
    replay_start = time.time()
    # Offset from original recording wall clock to replay wall clock; used to
    # rewrite ts_publish / ts_anchor consistently when not in keep-timestamps
    # mode.
    ts_offset_ms = int(replay_start * 1000) - base_recv

    published = 0
    for record in records:
        original_offset_ms = int(record["recv_ts_ms"]) - base_recv
        scheduled = replay_start + (original_offset_ms / 1000.0) / args.speed
        sleep_for = scheduled - time.time()
        if sleep_for > 0:
            time.sleep(sleep_for)

        payload = record["payload"]
        if not args.keep_timestamps:
            for key in ("ts_publish", "ts_anchor", "skin_temp_ts"):
                if key in payload and isinstance(payload[key], (int, float)):
                    if payload[key] > 0:
                        payload[key] = int(payload[key]) + ts_offset_ms
            ibi = payload.get("ibi") or {}
            if isinstance(ibi.get("ts"), list):
                ibi["ts"] = [int(t) + ts_offset_ms for t in ibi["ts"]]
            ppg = payload.get("ppg") or {}
            if isinstance(ppg.get("ts_ms"), list):
                ppg["ts_ms"] = [int(t) + ts_offset_ms for t in ppg["ts_ms"]]

        client.publish("biofizic/acquisition/batch", json.dumps(payload), qos=0)
        published += 1
        if published % 30 == 0:
            print(f"  replayed {published}/{len(records)}")

    client.loop_stop()
    client.disconnect()
    elapsed = time.time() - replay_start
    print(f"replayed {published} batches in {elapsed:.1f}s (speed={args.speed}x)")


if __name__ == "__main__":
    main()
