"""User emotion-feedback collection: a quadrant tap pairs the user's label with
the latest 33-feature PPG vector + each model's prediction, appended to a JSONL
store. Bad quadrants are ignored; a label with no fresh features is still stored
(absence of usable signal is itself informative)."""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import affectus.shared.feedback_store as fs
from affectus.shared.feedback_store import append_feedback, load_feedback
from services.compute_engine import ComputeEngineService


def _tmp() -> Path:
    return Path(tempfile.mkdtemp()) / "fb.jsonl"


class _FakeClient:
    def __init__(self):
        self.pub = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.pub.append((topic, json.loads(payload)))


def test_append_and_load_roundtrip():
    p = _tmp()
    feats = [float(i) * 0.1 for i in range(33)]
    row = append_feedback("Stresat", feats, arousal_z=1.2, features_age_s=1.5,
                          preds={"wesad": 0.3, "eevr": 0.6, "case": 0.66}, path=p)
    assert row["quadrant_code"] == 4
    assert row["n_features"] == 33
    rows = load_feedback(p)
    assert len(rows) == 1
    assert rows[0]["wesad_p_positive"] == 0.3
    assert rows[0]["features"][0] == 0.0


def test_label_without_features_still_stored():
    # A tap when no fresh PPG window exists must still record the label.
    p = _tmp()
    append_feedback("Calm", None, path=p)
    rows = load_feedback(p)
    assert len(rows) == 1
    assert rows[0]["n_features"] == 0
    assert rows[0]["features"] is None


def test_handle_feedback_pairs_label_with_features(monkeypatch):
    p = _tmp()
    monkeypatch.setattr(fs, "default_path", lambda: p)

    class FakeEng:
        last_features = [float(i) for i in range(33)]
        last_features_ts = int(time.time() * 1000) - 1500

    class FakeLegacy:
        _valence_wesad = FakeEng()

    svc = ComputeEngineService.__new__(ComputeEngineService)
    svc.legacy = FakeLegacy()
    svc._last_pred = {"wesad": 0.2, "eevr": 0.7, "case": 0.66, "arousal_z": 0.9, "ts": 1}
    client = _FakeClient()

    svc._handle_feedback(client, {"quadrant": "Bucuros", "ts": int(time.time() * 1000)}, time.time())

    rows = load_feedback(p)
    assert len(rows) == 1
    assert rows[0]["quadrant"] == "Bucuros"
    assert rows[0]["n_features"] == 33
    assert rows[0]["arousal_z"] == 0.9
    assert 1.0 <= rows[0]["features_age_s"] <= 2.5  # ~1.5 s old window
    # summary re-published for the dashboard
    assert any(t == "biofizic/legacy/feedback" for t, _ in client.pub)


def test_handle_feedback_ignores_bad_quadrant(monkeypatch):
    p = _tmp()
    monkeypatch.setattr(fs, "default_path", lambda: p)
    svc = ComputeEngineService.__new__(ComputeEngineService)
    svc.legacy = type("L", (), {"_valence_wesad": None})()
    svc._last_pred = {}
    svc._handle_feedback(_FakeClient(), {"quadrant": "nonsense"}, time.time())
    assert load_feedback(p) == []
