"""
Simulator MQTT biofizic/state — test Unity fără ceas.
Oprește emotion_classifier_v2 când rulezi simulatorul.
"""

from __future__ import annotations

import json
import time
from typing import Any

import paho.mqtt.client as mqtt
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Biofizic Simulator", version="2")

BROKER = "paxbespoke.automateflow.ro"
PORT = 1883


def _emotion_label(arousal_10: int) -> str:
    if arousal_10 <= 3:
        return "Calm"
    if arousal_10 <= 5:
        return "Activ"
    if arousal_10 <= 7:
        return "Ridicat"
    return "Intens"


def make_payload(arousal_10: int, valence_10: int = 5, **kw: Any) -> dict[str, Any]:
    a10 = max(0, min(10, arousal_10))
    v10 = max(0, min(10, valence_10))
    conf = float(kw.get("confidence", 0.92))
    trust = bool(kw.get("signal_trustworthy", True))
    ok = bool(kw.get("signal_ok", True))

    return {
        "ts": int(time.time() * 1000),
        "arousal": round(a10 / 10.0, 3),
        "arousal_10": a10,
        "arousal_rel": 0.0,
        "valence": round(v10 / 10.0, 3),
        "valence_10": v10,
        "confidence": round(conf, 3),
        "fresh_score": 1.0 if trust else 0.3,
        "stale": not trust,
        "signal_ok": ok,
        "signal_trustworthy": trust,
        "motion_gated": False,
        "emotion": kw.get("emotion") or _emotion_label(a10),
        "display_on": True,
        "hr": int(kw.get("mean_hr", 75)),
        "mean_hr": float(kw.get("mean_hr", 75.0)),
        "rmssd": float(kw.get("rmssd", 50.0)),
        "ibi_n": 80 if ok else 4,
        "window_sec": 60.0 if ok else 3.0,
        "sec_since_last_ibi": 1.0 if trust else 20.0,
        "profile_ready": True,
        "z_hr": 0.0,
        "z_rmssd": 0.0,
        "skin_trend": 0.0,
        "skin_trend_used": False,
        "lf_hf_weight": 0.0,
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
    valence_10: int = Field(5, ge=0, le=10)
    confidence: float = Field(0.92, ge=0.0, le=1.0)
    signal_trustworthy: bool = True


@app.post("/state")
def set_state(body: StateBody):
    payload = make_payload(
        body.arousal_10,
        body.valence_10,
        confidence=body.confidence,
        signal_trustworthy=body.signal_trustworthy,
        signal_ok=body.signal_trustworthy,
    )
    publish(payload)
    return {"sent": payload}


@app.post("/scene/calm")
def scene_calm():
    p = make_payload(2, 9, mean_hr=68.0, rmssd=65.0)
    publish(p)
    return {"scene": "calm", "sent": p}


@app.post("/scene/activ")
def scene_activ():
    p = make_payload(5, 5, mean_hr=78.0, rmssd=52.0)
    publish(p)
    return {"scene": "activ", "sent": p}


@app.post("/scene/ridicat")
def scene_ridicat():
    p = make_payload(7, 3, mean_hr=92.0, rmssd=28.0)
    publish(p)
    return {"scene": "ridicat", "sent": p}


@app.post("/scene/intens")
def scene_intens():
    p = make_payload(10, 1, mean_hr=98.0, rmssd=20.0)
    publish(p)
    return {"scene": "intens", "sent": p}
