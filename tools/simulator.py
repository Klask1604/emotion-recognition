"""
MQTT simulator for biofizic/state. Useful for testing the watch UI or Grafana
without a real Galaxy Watch attached. Stop the compute-engine container while
running this so its real decisions do not race with the simulated ones.
"""

from __future__ import annotations

import json
import time
from typing import Any

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Biofizic Simulator", version="3")

BROKER = "paxbespoke.automateflow.ro"
PORT = 1883


def _emotion_label(arousal_10: int) -> str:
    if arousal_10 <= 2:
        return "Relaxat"
    if arousal_10 <= 4:
        return "Echilibrat"
    if arousal_10 <= 6:
        return "Moderat"
    if arousal_10 <= 8:
        return "Alert"
    return "Ridicat"


def make_payload(arousal_10: int, **kw: Any) -> dict[str, Any]:
    a10 = max(0, min(10, arousal_10))
    conf = float(kw.get("confidence", 0.92))
    ok = bool(kw.get("signal_ok", True))

    return {
        "ts": int(time.time() * 1000),
        "engine": "simulator",
        "arousal_10": a10,
        "arousal_pct": round(a10 * 10.0, 1),
        "confidence": round(conf, 3),
        "signal_ok": ok,
        "motion_gated": False,
        "emotion": kw.get("emotion") or _emotion_label(a10),
        "emotion_baseline": kw.get("emotion_baseline") or _emotion_label(a10),
        "labels_agree": True,
        "display_on": True,
        "hr": int(kw.get("mean_hr", 75)),
        "mean_hr": float(kw.get("mean_hr", 75.0)),
        "rmssd": float(kw.get("rmssd", 50.0)),
        "stress_index": float(kw.get("stress_index", 12.0)),
        "baseline_si": float(kw.get("baseline_si", 12.0)),
        "z_si": float(kw.get("z_si", 0.0)),
        "profile_ready": True,
        "baseline_ready": True,
        "motion_state": kw.get("motion_state", "still"),
        "signal_quality": round(conf, 3),
        "artifact_rate": float(kw.get("artifact_rate", 0.0)),
        "motion_energy": float(kw.get("motion_energy", 0.0)),
        "alert": bool(kw.get("alert", False)),
    }


def publish(payload: dict[str, Any]) -> None:
    client = mqtt.Client(
        client_id="biofizic_simulator",
        callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
    )
    client.connect(BROKER, PORT)
    client.publish("biofizic/state", json.dumps(payload), qos=0)
    client.disconnect()


class StateBody(BaseModel):
    arousal_10: int = Field(5, ge=0, le=10)
    confidence: float = Field(0.92, ge=0.0, le=1.0)
    signal_ok: bool = True


@app.post("/state")
def set_state(body: StateBody):
    payload = make_payload(
        body.arousal_10,
        confidence=body.confidence,
        signal_ok=body.signal_ok,
    )
    publish(payload)
    return {"sent": payload}


@app.post("/scene/calm")
def scene_calm():
    p = make_payload(2, mean_hr=68.0, rmssd=65.0, stress_index=6.0)
    publish(p)
    return {"scene": "calm", "sent": p}


@app.post("/scene/activ")
def scene_activ():
    p = make_payload(5, mean_hr=78.0, rmssd=52.0, stress_index=11.0)
    publish(p)
    return {"scene": "activ", "sent": p}


@app.post("/scene/ridicat")
def scene_ridicat():
    p = make_payload(7, mean_hr=92.0, rmssd=28.0, stress_index=20.0)
    publish(p)
    return {"scene": "ridicat", "sent": p}


@app.post("/scene/intens")
def scene_intens():
    p = make_payload(10, mean_hr=98.0, rmssd=20.0, stress_index=32.0)
    publish(p)
    return {"scene": "intens", "sent": p}
