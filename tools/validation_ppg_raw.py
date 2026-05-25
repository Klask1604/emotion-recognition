#!/usr/bin/env python3
"""
Test 1 — Confirma ca batch-urile PPG raw ajung pe VPS.

Ruleaza pe VPS sau local:
    python validation_ppg_raw.py

Astepta-te la: batch-uri la ~2s, n~50, green non-zero si variabil.
"""

import json
import time

import paho.mqtt.client as mqtt

BROKER = "paxbespoke.automateflow.ro"
PORT   = 1883

batch_count = 0
start_ts    = time.time()


def on_connect(client, userdata, flags, rc, props=None):
    if rc == 0:
        print(f"[OK] Conectat la {BROKER}:{PORT}")
        client.subscribe("biofizic/ppg/raw", qos=0)
    else:
        print(f"[ERR] Conectare esuata rc={rc}")


def on_message(client, userdata, msg):
    global batch_count
    try:
        data = json.loads(msg.payload)
    except Exception as e:
        print(f"[ERR] JSON invalid: {e}")
        return

    batch_count += 1
    n         = data.get("n", 0)
    ts_list   = data.get("ts", [])
    green     = data.get("green", [])
    ir        = data.get("ir", [])
    fs        = data.get("fs", "?")
    ts_start  = data.get("ts_start", 0)

    span_ms   = (ts_list[-1] - ts_list[0]) if len(ts_list) > 1 else 0
    g_min     = min(green) if green else 0
    g_max     = max(green) if green else 0
    g_range   = g_max - g_min
    ir_range  = (max(ir) - min(ir)) if ir else 0

    elapsed = time.time() - start_ts

    # Samsung GW7 batchuieste PPG intern la ~8s (200 samples × 40ms).
    # Rata reala = n / (span_ms / 1000.0) — trebuie sa fie ~25 Hz.
    rate_real = n / (span_ms / 1000.0) if span_ms > 0 else 0.0
    status = []
    if n < 10:
        status.append(f"WARN n={n} prea mic — PPG posibil oprit")
    if span_ms > 0 and not (1000 <= span_ms <= 16000):
        status.append(f"WARN span={span_ms}ms neasteptat (normal: 2000–10000ms)")
    if span_ms > 0 and not (20 <= rate_real <= 30):
        status.append(f"WARN fs_real={rate_real:.1f}Hz (asteptat ~25 Hz)")
    if g_range < 10:
        status.append("WARN green variatie mica — PPG posibil oprit")
    if not status:
        status.append(f"OK  fs_real={rate_real:.1f}Hz")

    print(
        f"[{elapsed:6.1f}s] batch#{batch_count:3d}  n={n:3d}  span={span_ms:5d}ms  "
        f"fs={fs}  green=[{g_min},{g_max}] range={g_range}  ir_range={ir_range}  "
        f"  {' | '.join(status)}"
    )


def main():
    print(f"Ascult pe biofizic/ppg/raw ({BROKER}) — Ctrl+C pentru a opri\n")
    c = mqtt.Client(
        client_id="biofizic_ppg_validator",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    c.on_connect = on_connect
    c.on_message = on_message
    c.connect(BROKER, PORT, keepalive=60)
    try:
        c.loop_forever()
    except KeyboardInterrupt:
        elapsed = time.time() - start_ts
        rate = batch_count / elapsed if elapsed > 0 else 0
        print(f"\n[FINAL] {batch_count} batch-uri in {elapsed:.0f}s (~{rate:.2f}/s)")


if __name__ == "__main__":
    main()
