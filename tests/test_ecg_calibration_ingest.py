"""ECG calibration ingestion: raw ECG chunks -> R-peaks -> IBI -> the arousal
baseline locks (motion-gate-bypassed) and the server publishes phase="done"."""

from __future__ import annotations

import json
import math

import pytest

pytest.importorskip("scipy")

from affectus.engine.pipeline import PhysiologyPipeline
from services.compute_engine import ComputeEngineService


class _FakeClient:
    def __init__(self):
        self.published = []

    def publish(self, topic, payload, qos=0, retain=False):
        self.published.append((topic, json.loads(payload)))


def _service(tmp_path):
    svc = ComputeEngineService.__new__(ComputeEngineService)
    # Build a real pipeline but point its persisted stores at tmp_path so the test
    # is isolated from the host's data/ files.
    from affectus.dsp.baseline import RestBaselineStore
    svc.pipeline = PhysiologyPipeline()
    svc.pipeline.baseline = RestBaselineStore(path=tmp_path / "rest.json")
    svc._calibrating = True
    svc._baseline_was_ready = False
    svc._ecg_cal_mv = []
    svc._ecg_cal_ts = []
    svc._ecg_cal_lead = []
    svc._ecg_last_peak_ms = 0
    return svc


def _ecg_chunk(start_ms: int, seconds: float, fs=500, bpm=75):
    """One ECG chunk: Gaussian QRS bumps at `bpm`, lead on."""
    rr_ms = 60_000 / bpm
    n = int(seconds * fs)
    samples = []
    # bump centres relative to chunk start
    t = 0.0
    centres = []
    while t < seconds * 1000:
        centres.append(t)
        t += rr_ms
    for i in range(n):
        ms = i * 1000.0 / fs
        v = 0.0
        for c in centres:
            d = (ms - c) * fs / 1000.0
            if abs(d) < 8:
                v += 1000.0 * math.exp(-(d * d) / 4.0)
        samples.append({"ts": int(start_ms + ms), "ecg_mv": int(v), "lead_off": 0})
    return {"recv_ms": start_ms, "samples": samples, "final": False}


def test_ecg_calibration_locks_baseline_and_publishes_done(tmp_path):
    svc = _service(tmp_path)
    client = _FakeClient()
    assert not svc.pipeline.baseline.is_ready

    # Feed several spaced ECG chunks (each ~6 s of clean 75 bpm ECG), advancing
    # `now` by >3 s between chunks so the baseline's spacing gate accepts them.
    base_ms = 1_000_000
    now = 1000.0
    for k in range(12):
        chunk = _ecg_chunk(base_ms + k * 4000, seconds=6.0)
        svc._handle_ecg_calibration(client, chunk, now=now)
        now += 4.0

    assert svc.pipeline.baseline.is_ready, "ECG IBI should lock the baseline"
    done = [p for (t, p) in client.published
            if t == "biofizic/calibration/status" and p.get("phase") == "done"]
    assert done, "server should publish phase=done when ECG locks the baseline"


def test_ecg_ignored_when_not_calibrating(tmp_path):
    svc = _service(tmp_path)
    svc._calibrating = False
    client = _FakeClient()
    svc._handle_ecg_calibration(client, _ecg_chunk(1_000_000, 6.0), now=1000.0)
    assert svc._ecg_cal_mv == []          # nothing buffered outside calibration
    assert client.published == []


def test_ecg_no_duplicate_beats(tmp_path):
    # Re-detecting over a growing buffer must NOT double-count: each R-peak is
    # accumulated into the ECG beat list exactly once (the source-side dedup).
    svc = _service(tmp_path)
    client = _FakeClient()
    svc._handle_ecg_calibration(client, _ecg_chunk(1_000_000, 6.0), now=1000.0)
    after_first = len(svc.pipeline._ecg_cal_beats)
    svc._handle_ecg_calibration(client, _ecg_chunk(1_006_000, 4.0), now=1004.0)
    after_second = len(svc.pipeline._ecg_cal_beats)
    # ~75 bpm over 10 s ≈ 12 beats; with dedup the count grows monotonically and
    # stays near the true beat count, not ~2x it.
    assert after_second > after_first
    assert after_second <= 16, f"too many beats ({after_second}) — duplicates leaked"


def test_ecg_lock_bounded_by_real_time(tmp_path):
    # The baseline must NOT lock from a single huge chunk: the spacing gate ties
    # the 8 required samples to real elapsed seconds (one accepted ~per second).
    svc = _service(tmp_path)
    client = _FakeClient()
    # One 30 s chunk delivered at a single instant -> at most 1 spaced sample,
    # nowhere near the 8 needed, so it must not be ready yet.
    svc._handle_ecg_calibration(client, _ecg_chunk(2_000_000, 30.0), now=5000.0)
    assert not svc.pipeline.baseline.is_ready
    # Now advance real time across spaced calls -> it locks within ~8 s.
    for k in range(10):
        svc._handle_ecg_calibration(client, _ecg_chunk(2_030_000 + k * 1000, 1.5), now=5001.0 + k)
    assert svc.pipeline.baseline.is_ready
