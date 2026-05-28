"""Smoke tests for the cardiac comparator (services/test_engine.py).

These do not touch MQTT — they exercise the buffers + DSP pipeline on
synthetic inputs so we know peak detection -> HRV math returns sensible
numbers per source. The integration with paho is covered by running the
service against the broker in production.
"""

from __future__ import annotations

import math

from biofizic.engine.pipeline import PhysiologyPipeline

from services.test_engine import (
    WINDOW_SEC,
    CardiacComparator,
    InMemoryBaselineStore,
    _IbiBuffer,
    _LastBatchContext,
    _PpgBuffer,
    _normalize_ts_ms,
    _walk_back_ibi_timestamps,
)


def _synthetic_ppg(duration_s: float, fs_hz: int, hr_bpm: float = 72.0):
    """Clean sinusoidal PPG at heart rate hr_bpm — what the on-demand tracker
    should look like at rest, no noise, no motion."""
    n = int(duration_s * fs_hz)
    freq = hr_bpm / 60.0
    ts = list(range(n))
    greens = [
        int(2048 + 500 * math.sin(2 * math.pi * freq * t / fs_hz))
        for t in range(n)
    ]
    ts_ms = [int(t * 1000 / fs_hz) for t in ts]
    return greens, ts_ms


def test_normalize_ts_handles_ns_and_ms():
    assert _normalize_ts_ms(0) == 0
    assert _normalize_ts_ms(None) == 0
    assert _normalize_ts_ms(1_700_000_000_000) == 1_700_000_000_000  # ms epoch
    assert _normalize_ts_ms(1_700_000_000_000_000_000) == 1_700_000_000_000  # ns -> ms


def test_ppg_buffer_trims_older_than_window():
    buf = _PpgBuffer()
    base = 1_700_000_000_000
    # Two old samples (40 s ago, outside the 30 s window) + one fresh.
    buf.add(base, 100)
    buf.add(base + 5_000, 110)
    buf.add(base + 40_000, 200)
    greens, ts = buf.snapshot()
    # The 40 s old samples must be evicted by the time the newest one lands.
    assert ts[-1] == base + 40_000
    assert all(t >= ts[-1] - WINDOW_SEC * 1000 for t in ts)


def test_walk_back_ibi_timestamps_reconstructs_anchor():
    intervals = [800, 820, 810]
    anchor = 1_700_000_010_000
    entries = _walk_back_ibi_timestamps(intervals, anchor)
    assert entries[-1].timestamp_ms == anchor
    assert entries[1].timestamp_ms == anchor - 810
    assert entries[0].timestamp_ms == anchor - 810 - 820
    assert [e.interval_ms for e in entries] == intervals


def test_ibi_buffer_trims_older_than_window():
    buf = _IbiBuffer()
    intervals = [800, 800, 800, 800]
    # Anchor: first beat very old, last beat fresh.
    anchor_old = 1_700_000_000_000
    buf.extend(_walk_back_ibi_timestamps(intervals, anchor_old))
    # New burst 40 s later — should evict the old beats.
    buf.extend(_walk_back_ibi_timestamps(intervals, anchor_old + 40_000))
    snap = buf.snapshot()
    newest = snap[-1].timestamp_ms
    assert all((e.timestamp_ms or 0) >= newest - WINDOW_SEC * 1000 for e in snap)


def _patched_comparator(captured: list) -> CardiacComparator:
    """A comparator whose client.publish is captured into a list, so we can
    drive _publish_*_source without an MQTT broker."""

    class _FakeClient:
        def publish(self, topic, payload, qos=0):
            captured.append((topic, payload))

    comparator = CardiacComparator.__new__(CardiacComparator)
    comparator.buf_ppg_ond = _PpgBuffer()
    comparator.buf_ppg_cont = _PpgBuffer()
    comparator.buf_hr_ibi = _IbiBuffer()
    comparator.ppg_ond_pipeline = PhysiologyPipeline()
    comparator.ppg_ond_pipeline.baseline = InMemoryBaselineStore()
    comparator.ppg_cont_pipeline = PhysiologyPipeline()
    comparator.ppg_cont_pipeline.baseline = InMemoryBaselineStore()
    comparator._last_batch = _LastBatchContext()
    comparator._ppg_only_seq = 0
    comparator.client = _FakeClient()
    return comparator


def test_ppg_source_publishes_hr_near_72_bpm_on_clean_signal():
    captured: list = []
    comparator = _patched_comparator(captured)
    greens, ts_ms = _synthetic_ppg(duration_s=30.0, fs_hz=100, hr_bpm=72.0)
    for g, t in zip(greens, ts_ms):
        comparator.buf_ppg_ond.add(t, g)

    comparator._publish_ppg_source("ppg_ondemand", comparator.buf_ppg_ond, ts_ms=ts_ms[-1])

    assert captured, "expected at least one derived publish"
    topic, body = captured[-1]
    assert topic == "biofizic/test/derived/ppg_ondemand"
    import json as _json

    payload = _json.loads(body)
    assert payload["source"] == "ppg_ondemand"
    assert payload["peak_count"] >= 15  # ~36 beats in 30 s, allow slack
    assert 65 <= payload["hr_bpm"] <= 80, payload  # 72 bpm target
    assert payload["rmssd_ms"] >= 0  # clean sinusoid -> tiny but valid


def test_ppg_only_pipeline_publishes_state_with_source_tag():
    """The PPG-only PhysiologyPipeline tick should always publish a state
    payload (even if baseline is not yet ready) so InfluxDB / Grafana never
    have a column with zero data. The payload carries the source tag for
    Grafana GROUP BY filtering."""
    captured: list = []
    comparator = _patched_comparator(captured)

    # Seed _last_batch as if biofizic/acquisition/batch arrived (still, no motion).
    lb = comparator._last_batch
    lb.seen = True
    lb.ts_anchor_ms = 1_700_000_030_000
    lb.acc_band_cardiac = 0.005  # quiet wrist

    greens, ts_ms = _synthetic_ppg(duration_s=30.0, fs_hz=100, hr_bpm=72.0)
    for g, t in zip(greens, ts_ms):
        # Shift PPG timestamps so they end near ts_anchor (mimicking arrival).
        comparator.buf_ppg_ond.add(lb.ts_anchor_ms - (ts_ms[-1] - t), g)

    comparator._publish_ppg_only_source(
        "ppg_only_ondemand",
        comparator.buf_ppg_ond,
        comparator.ppg_ond_pipeline,
        ts_ms=lb.ts_anchor_ms,
    )

    assert captured, "expected at least one ppg_only state publish"
    topic, body = captured[-1]
    assert topic == "biofizic/test/derived/ppg_only_ondemand"
    import json as _json

    payload = _json.loads(body)
    assert payload["source"] == "ppg_only_ondemand"
    assert payload["engine"] == "test_ppg_only"
    # Baseline is cold; decision may be None, but the pipeline status fields
    # must be populated regardless so Grafana panels stay continuous.
    assert "signal_quality" in payload
    assert "ibi_buffer_size" in payload


def test_hr_source_publishes_when_buffer_has_two_beats():
    captured: list = []
    comparator = _patched_comparator(captured)
    intervals = [820] * 20  # 20 beats at ~73 bpm
    entries = _walk_back_ibi_timestamps(intervals, anchor_ts_ms=1_700_000_000_000)
    comparator.buf_hr_ibi.extend(entries)

    comparator._publish_hr_source(ts_ms=1_700_000_000_000)

    assert captured, "expected derived/hr_continuous publish"
    topic, body = captured[-1]
    assert topic == "biofizic/test/derived/hr_continuous"
    import json as _json

    payload = _json.loads(body)
    assert payload["source"] == "hr_continuous"
    assert 70 <= payload["hr_bpm"] <= 76
    assert payload["ibi_count"] == 20
