"""The compute engine's hello handler builds the right ack payload and caches
capabilities, without needing a live broker (a fake client captures publishes)."""

from __future__ import annotations

import json

import pytest


class _FakeClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0):
        self.published.append((topic, json.loads(payload), qos))


@pytest.fixture
def engine():
    # Import lazily so test collection doesn't require the MQTT stack to connect.
    from services.compute_engine import ComputeEngineService
    return ComputeEngineService(broker="localhost", port=1883)


def test_valid_announce_acks_ok_and_caches(engine):
    client = _FakeClient()
    engine._handle_hello(client, {
        "client_id": "gw7", "schema": 2,
        "capabilities": ["ibi", "hr", "ppg", "motion", "temp"],
    })
    topic, payload, qos = client.published[-1]
    assert topic == "biofizic/hello/ack"
    assert payload["status"] == "ok"
    assert payload["modules_active"] == ["hrv", "hr", "temp"]
    assert "gw7" in engine._announced_caps


def test_skin_temp_only_acks_error_and_no_cache(engine):
    client = _FakeClient()
    engine._handle_hello(client, {"client_id": "bad", "capabilities": ["temp"]})
    topic, payload, _ = client.published[-1]
    assert topic == "biofizic/hello/ack"
    assert payload["status"] == "error"
    assert payload["modules_active"] == []
    assert "bad" not in engine._announced_caps
