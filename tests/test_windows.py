"""Rolling buffers: IBI ingest must not drop beats; sensor median window."""

from __future__ import annotations

from biofizic.ingestion.messages import IbiBatchMessage
from biofizic.compute_features.windows import RollingIbiBuffer, RollingSensorBuffer


def test_ingest_keeps_all_beats_when_timestamps_partial():
    # 4 intervals but only 2 timestamps: the old zip() truncated to 2 beats.
    batch = IbiBatchMessage(
        timestamp_ms=10_000,
        intervals_ms=[800, 810, 790, 805],
        timestamps_ms=[9_200, 10_000],
    )
    buf = RollingIbiBuffer()
    buf.ingest_batch(batch)
    entries = buf.entries_in_last_ms(120_000, end_ms=10_000)
    assert len(entries) == 4, "all four beats must be retained"
    # Reconstructed timestamps are monotonic and end at the anchor.
    assert entries[-1].timestamp_ms == 10_000


def test_ingest_pairs_directly_when_timestamps_complete():
    batch = IbiBatchMessage(
        timestamp_ms=10_000,
        intervals_ms=[800, 810],
        timestamps_ms=[9_190, 10_000],
    )
    buf = RollingIbiBuffer()
    buf.ingest_batch(batch)
    entries = buf.entries_in_last_ms(120_000, end_ms=10_000)
    assert [e.timestamp_ms for e in entries] == [9_190, 10_000]


def test_sensor_buffer_window_median():
    buf = RollingSensorBuffer()
    for i, v in enumerate([0.1, 0.2, 0.3, 9.0]):
        buf.ingest(10_000 + i * 1_000, v)
    # All four within the 30 s window -> median of [0.1,0.2,0.3,9.0] (upper-mid).
    assert buf.median_in_last_ms(30_000, end_ms=13_000) == 0.3
