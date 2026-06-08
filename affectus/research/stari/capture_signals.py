"""Logger de captura pentru figurile de procesare de semnal din licenta.

Se aboneaza la magistrala MQTT si salveaza BRUT, fara nicio prelucrare, cadrele
de achizitie si segmentele PPG on-demand venite de pe ceas. Fiecare linie din
fisierul .jsonl este {recv_ms, topic, payload}, unde payload este JSON-ul exact
publicat de ceas. Plotarea (aliniere, PPG->filtrat->varfuri, artefacte, HRV) se
face OFFLINE din acest fisier, ca sa nu atingem deloc pipeline-ul live.

Rulare (laptop, broker remote al ceasului):
    venv/Scripts/python.exe -m affectus.research.stari.capture_signals \
        --broker paxbespoke.automateflow.ro --port 1883 --seconds 180 \
        --out data/capture_licenta.jsonl

Pe localhost (engine + ceas pe aceeasi retea): --broker localhost.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import paho.mqtt.client as mqtt

TOPICS = [
    ("biofizic/in/acquisition", 0),  # sensor frames + PPG 100 Hz
    ("biofizic/out/arousal", 0),     # the verdict, for the HRV/baseline-over-time figure
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--broker", default="localhost")
    ap.add_argument("--port", type=int, default=1883)
    ap.add_argument("--seconds", type=int, default=180, help="durata capturii")
    ap.add_argument("--out", default="data/capture_licenta.jsonl")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fh = out.open("w", encoding="utf-8")
    counts: dict[str, int] = {}

    def on_connect(client, userdata, flags, rc, properties=None):
        print(f"[capture] conectat (rc={rc}), abonare la {len(TOPICS)} topicuri")
        client.subscribe(TOPICS)

    def on_message(client, userdata, msg):
        rec = {
            "recv_ms": int(time.time() * 1000),
            "topic": msg.topic,
            "payload": msg.payload.decode("utf-8", "replace"),
        }
        fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh.flush()
        counts[msg.topic] = counts.get(msg.topic, 0) + 1

    client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
    client.on_connect = on_connect
    client.on_message = on_message
    client.connect(args.broker, args.port, 20)
    client.loop_start()

    t0 = time.time()
    try:
        while time.time() - t0 < args.seconds:
            elapsed = int(time.time() - t0)
            tot = sum(counts.values())
            print(f"\r[capture] {elapsed:3d}/{args.seconds}s  mesaje={tot}  {counts}", end="")
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n[capture] oprit manual")
    finally:
        client.loop_stop()
        client.disconnect()
        fh.close()
        print(f"\n[capture] gata -> {out}  ({sum(counts.values())} mesaje)")
        for t, n in sorted(counts.items()):
            print(f"   {t}: {n}")


if __name__ == "__main__":
    main()
