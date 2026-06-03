"""Pipeline ingest: a watch batch feeds the rolling buffers directly (no
intermediate SensorFrame). The batch IS the in-process sensor record."""

from __future__ import annotations

from affectus.io.messages import AcquisitionBatchMessage
from affectus.engine.pipeline import PhysiologyPipeline


def _batch(**kw) -> AcquisitionBatchMessage:
    base = dict(
        timestamp_publish_ms=1000, timestamp_anchor_ms=1810, sequence=1,
        heart_rate_bpm=72.0, skin_temperature_c=33.0, ambient_temperature_c=24.0,
        acc_band_cardiac=0.01, ibi_intervals_ms=[820, 810], ibi_timestamps_ms=[1000, 1810],
    )
    base.update(kw)
    return AcquisitionBatchMessage(**base)


def test_ingest_feeds_ibi_buffer():
    p = PhysiologyPipeline()
    p.ingest_acquisition(_batch())
    # both beats landed in the rolling IBI buffer
    assert p.ibi_buffer.entries_in_last_ms(120_000, end_ms=1810)


def test_ingest_records_batch_as_sensor():
    p = PhysiologyPipeline()
    b = _batch()
    p.ingest_acquisition(b)
    # the batch itself is the sensor record the decision reads
    assert p.state.last_sensor is b
    assert p.state.last_sensor.heart_rate_bpm == 72.0
    assert p.state.last_sensor.skin_temperature_c == 33.0


def test_ingest_empty_ibi_is_noop_for_buffer():
    p = PhysiologyPipeline()
    p.ingest_acquisition(_batch(ibi_intervals_ms=[], ibi_timestamps_ms=[]))
    # no beats -> buffer empty, but the batch is still recorded as the sensor
    assert not p.ibi_buffer.entries_in_last_ms(120_000, end_ms=1810)
    assert p.state.last_sensor is not None
