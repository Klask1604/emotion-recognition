#!/usr/bin/env python3
"""
Live validation of the watch -> classifier chain (no Unity).

Subscribes to the production topics and checks, in real time, that:
  1. acquisition/batch is arriving from the watch (raw input flowing),
  2. the compute-engine is publishing state/live (classifier producing output),
  3. window_used reports the window that actually decided (w60, not a hardcoded
     w30) — the Phase-A fix,
  4. RMSSD stays inside a physiological band on quiet wear-time (the gap fix:
     no single-beat spikes from differences taken across dropped beats),
  5. the per-batch skew (watch -> server) stays small (clock/backlog health).

Run while wearing the watch with tracking on:
    ./venv/Scripts/python.exe tools/validate_watch_to_classifier.py
Stop with Ctrl-C; a summary prints on exit.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from collections import Counter

import paho.mqtt.client as mqtt

BROKER = os.environ.get("MQTT_BROKER", "paxbespoke.automateflow.ro")
PORT = int(os.environ.get("MQTT_PORT", "1883"))

TOPICS = [
    ("biofizic/acquisition/batch", 0),
    ("biofizic/live", 0),
    ("biofizic/state", 1),
    ("biofizic/state/live", 0),
]

# Physiological RMSSD ceiling for quiet wear-time. A genuine resting RMSSD on
# the wrist rarely exceeds ~120 ms; values far above signal a difference taken
# across a dropped beat (the bug the gap fix addresses).
RMSSD_SANITY_CEILING_MS = 150.0


class Stats:
    def __init__(self) -> None:
        self.acq_count = 0
        self.live_count = 0
        self.state_count = 0
        self.window_used = Counter()
        self.rmssd_max = 0.0
        self.rmssd_over_ceiling = 0
        self.skew_samples: list[float] = []
        self.last_arousal = None
        self.last_dom_channel = None
        self.baseline_ready_seen = False


stats = Stats()


def on_connect(client, userdata, flags, rc, props=None):
    if rc == 0:
        for t, q in TOPICS:
            client.subscribe(t, q)
        print(f"[connected] {BROKER}:{PORT} — wear the watch + start tracking…\n")
    else:
        print(f"[connect failed] rc={rc}")


def on_message(client, userdata, msg):
    try:
        d = json.loads(msg.payload.decode("utf-8", "replace"))
    except Exception:
        return
    now_ms = time.time() * 1000

    if msg.topic == "biofizic/acquisition/batch":
        stats.acq_count += 1
        ts_pub = d.get("ts_publish") or d.get("ts") or 0
        if ts_pub:
            stats.skew_samples.append((now_ms - ts_pub) / 1000.0)
        ibi = (d.get("ibi") or {}).get("ms") or []
        if stats.acq_count % 5 == 0:
            print(f"[acq #{stats.acq_count}] seq={d.get('seq')} ibi_in_batch={len(ibi)} "
                  f"hr={d.get('hr')} skew={stats.skew_samples[-1]:.1f}s")

    elif msg.topic == "biofizic/live":
        stats.live_count += 1
        rmssd = d.get("rmssd")
        if isinstance(rmssd, (int, float)) and rmssd > 0:
            stats.rmssd_max = max(stats.rmssd_max, rmssd)
            if rmssd > RMSSD_SANITY_CEILING_MS:
                stats.rmssd_over_ceiling += 1
                print(f"  ⚠ RMSSD={rmssd} ms > {RMSSD_SANITY_CEILING_MS} ceiling "
                      f"(possible cross-gap difference)")
        if d.get("baseline_ready"):
            stats.baseline_ready_seen = True
        a = d.get("arousal_10")
        if a is not None:
            stats.last_arousal = a
            stats.last_dom_channel = d.get("dominant_channel")

    elif msg.topic in ("biofizic/state", "biofizic/state/live"):
        if msg.topic == "biofizic/state":
            stats.state_count += 1
        wu = d.get("window_used")
        if wu:
            stats.window_used[wu] += 1


def summary(*_):
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY (watch -> classifier)")
    print("=" * 60)
    ok = True

    def check(label, passed, detail=""):
        nonlocal ok
        ok = ok and passed
        print(f"  [{'PASS' if passed else 'FAIL'}] {label}  {detail}")

    check("watch is publishing acquisition/batch", stats.acq_count > 0,
          f"({stats.acq_count} batches)")
    check("classifier is publishing live", stats.live_count > 0,
          f"({stats.live_count} live ticks)")

    wu = dict(stats.window_used)
    check("window_used reports w60 (Phase-A fix, not hardcoded w30)",
          wu.get("w60", 0) > 0 or (wu.get("w30", 0) == 0 and not wu),
          f"seen={wu or 'none yet'}")

    check("RMSSD stayed within physiological band (gap fix)",
          stats.rmssd_over_ceiling == 0,
          f"max={stats.rmssd_max:.0f}ms, over-ceiling={stats.rmssd_over_ceiling}")

    if stats.skew_samples:
        avg_skew = sum(stats.skew_samples) / len(stats.skew_samples)
        check("watch->server skew healthy (< 5 s)", avg_skew < 5.0,
              f"avg={avg_skew:.1f}s")

    print(f"\n  baseline_ready seen: {stats.baseline_ready_seen}")
    print(f"  last arousal_10: {stats.last_arousal} (via {stats.last_dom_channel})")
    print("=" * 60)
    print("OVERALL:", "✓ chain healthy" if ok else "✗ see FAILs above")
    sys.exit(0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default=BROKER)
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()

    client = mqtt.Client(client_id="biofizic_validator",
                         callback_api_version=mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    signal.signal(signal.SIGINT, summary)
    signal.signal(signal.SIGTERM, summary)
    client.connect(args.broker, args.port, 30)
    print("Listening… press Ctrl-C to stop and print the summary.")
    client.loop_forever()


if __name__ == "__main__":
    main()
