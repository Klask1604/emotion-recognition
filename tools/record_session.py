#!/usr/bin/env python3
"""
Subscribe to biofizic/acquisition/batch and write every payload as a JSONL
line. Each line is augmented with the local receive time so the replayer can
honour original cadence.

Usage:
    python tools/record_session.py --output session_2026-05-26.jsonl --duration 300

Stops automatically after --duration seconds, or on Ctrl-C.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from pathlib import Path

import paho.mqtt.client as mqtt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    parser.add_argument("--output", required=True, help="JSONL file to write")
    parser.add_argument(
        "--duration",
        type=int,
        default=300,
        help="Recording length in seconds (0 = until Ctrl-C)",
    )
    args = parser.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fh = out_path.open("w", encoding="utf-8")
    count = 0
    started_at = time.time()

    def on_connect(client, userdata, flags, rc, props=None):
        if rc == 0:
            client.subscribe("biofizic/acquisition/batch", qos=0)
            print(f"connected to {args.broker}:{args.port}, recording to {out_path}")
        else:
            print(f"connect failed rc={rc}", file=sys.stderr)

    def on_message(client, userdata, msg):
        nonlocal count
        try:
            payload = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except json.JSONDecodeError:
            return
        record = {"recv_ts_ms": int(time.time() * 1000), "payload": payload}
        fh.write(json.dumps(record) + "\n")
        fh.flush()
        count += 1
        if count % 30 == 0:
            print(f"  recorded {count} batches")

    client = mqtt.Client(
        client_id="biofizic_recorder",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, keepalive=60)

    def shutdown(_a, _b):
        client.disconnect()
        fh.close()
        elapsed = time.time() - started_at
        print(f"stopped after {elapsed:.1f}s, {count} batches written to {out_path}")
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    client.loop_start()
    try:
        if args.duration > 0:
            time.sleep(args.duration)
        else:
            while True:
                time.sleep(1)
    finally:
        shutdown(None, None)


if __name__ == "__main__":
    main()
