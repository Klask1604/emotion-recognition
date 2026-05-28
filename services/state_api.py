"""
Manual state publisher for VR testing — FastAPI over MQTT.

When the classifier (compute-engine) is OFF, this API publishes a message in the
exact `biofizic/state` schema the Unity scene consumes, so you can drive arousal
state transitions by hand (HTTP) instead of from the watch.

Run:
    uvicorn services.state_api:app --host 0.0.0.0 --port 8200
    # broker via env: MQTT_BROKER / MQTT_PORT (default paxbespoke.automateflow.ro:1883)

Unity drives its mood off arousal_10 with a 3-epoch confirmation streak, so each
call publishes `repeat` (default 3) identical messages to commit a band change.

Endpoints (see GET / for the live list):
    POST /state            full control (JSON body)
    POST /arousal/{1..10}  quick set arousal, auto emotion label
    POST /preset/{name}    calm | balanced | elevated | intense
    POST /clear            clear the retained state message
    GET  /health
"""

from __future__ import annotations

import asyncio
import os
import time

import paho.mqtt.client as mqtt
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BROKER = os.environ.get("MQTT_BROKER", "paxbespoke.automateflow.ro")
PORT = int(os.environ.get("MQTT_PORT", "1883"))
TOPIC = "biofizic/state"

# Matches biofizic.engine.arousal_mapper.arousal_scale_10_to_label.
def emotion_label(arousal_10: int) -> str:
    if arousal_10 <= 2:
        return "Relaxat"
    if arousal_10 <= 4:
        return "Echilibrat"
    if arousal_10 <= 6:
        return "Moderat"
    if arousal_10 <= 8:
        return "Alert"
    return "Ridicat"


# Preset name -> arousal_10, aligned with Unity's mood bands
# (calm 1-3 / balanced 4-5 / elevated 6-7 / intense 8-10).
PRESETS = {"calm": 2, "balanced": 5, "elevated": 7, "intense": 9}


class StateRequest(BaseModel):
    arousal_10: int = Field(ge=1, le=10)
    confidence: float = Field(default=0.9, ge=0.0, le=1.0)
    emotion: str | None = None  # auto from arousal if omitted
    motion_state: str = "still"  # "still" | "moving"
    dominant_channel: str = "hrv"  # hrv | hr | blend | none
    baseline_ready: bool = True
    repeat: int = Field(default=3, ge=1, le=20)  # >=3 to commit a Unity band change
    interval_s: float = Field(default=0.4, ge=0.0, le=5.0)


client = mqtt.Client(
    client_id="biofizic_state_api",
    callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
)

app = FastAPI(title="Biofizic State API", version="1.0")


@app.on_event("startup")
def _connect() -> None:
    client.connect(BROKER, PORT, keepalive=30)
    client.loop_start()


@app.on_event("shutdown")
def _disconnect() -> None:
    client.loop_stop()
    client.disconnect()


def _build_payload(req: StateRequest) -> dict:
    a = req.arousal_10
    label = req.emotion or emotion_label(a)
    return {
        "ts": int(time.time() * 1000),
        "engine": "state_api",
        "live": False,
        "arousal_10": a,
        "arousal_pct": round(a * 10.0, 1),
        "emotion": label,
        "emotion_baseline": label,
        "labels_agree": True,
        "confidence": round(req.confidence, 3),
        "dominant_channel": req.dominant_channel,
        "motion_state": req.motion_state,
        "signal_quality": round(req.confidence, 3),
        "baseline_ready": req.baseline_ready,
        "profile_ready": req.baseline_ready,
        "why": "manual override (state_api)",
    }


async def _publish(req: StateRequest) -> dict:
    import json

    payload = _build_payload(req)
    body = json.dumps(payload)
    for i in range(req.repeat):
        # Retain so a freshly-connecting Unity client gets the last state too.
        client.publish(TOPIC, body, qos=1, retain=True)
        if i < req.repeat - 1 and req.interval_s > 0:
            await asyncio.sleep(req.interval_s)
    return {"published": req.repeat, "topic": TOPIC, "payload": payload}


@app.get("/")
def root() -> dict:
    return {
        "service": "Biofizic State API",
        "broker": f"{BROKER}:{PORT}",
        "topic": TOPIC,
        "presets": PRESETS,
        "endpoints": {
            "POST /state": "full control (StateRequest body)",
            "POST /arousal/{1..10}": "quick set arousal",
            "POST /preset/{name}": list(PRESETS.keys()),
            "POST /clear": "clear retained state",
            "GET /health": "mqtt connection status",
        },
        "note": "Unity needs 3 identical epochs to change band; repeat>=3 (default).",
    }


@app.get("/health")
def health() -> dict:
    return {"connected": client.is_connected(), "broker": f"{BROKER}:{PORT}"}


@app.post("/state")
async def post_state(req: StateRequest) -> dict:
    return await _publish(req)


@app.post("/arousal/{value}")
async def post_arousal(value: int) -> dict:
    if not 1 <= value <= 10:
        raise HTTPException(400, "arousal must be 1..10")
    return await _publish(StateRequest(arousal_10=value))


@app.post("/preset/{name}")
async def post_preset(name: str) -> dict:
    key = name.lower()
    if key not in PRESETS:
        raise HTTPException(400, f"unknown preset; use one of {list(PRESETS)}")
    return await _publish(StateRequest(arousal_10=PRESETS[key]))


@app.post("/clear")
def clear() -> dict:
    # Empty retained message clears the broker's retained state.
    client.publish(TOPIC, payload=b"", qos=1, retain=True)
    return {"cleared": TOPIC}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("STATE_API_PORT", "8200")))
