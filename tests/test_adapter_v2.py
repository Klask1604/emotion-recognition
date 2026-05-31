"""The schema-v2 adapter: a watch batch maps to a SensorFrame with the right
capabilities and the same data, and the pipeline ingest path is equivalent."""

from __future__ import annotations

from affectus.ingestion.messages import AcquisitionBatchMessage
from affectus.devices.wrist.adapter import frame_from_acquisition_v2
from affectus.contract.capabilities import Capability


def _batch(**kw) -> AcquisitionBatchMessage:
    base = dict(
        timestamp_publish_ms=1000, timestamp_anchor_ms=1000, sequence=1,
        heart_rate_bpm=72.0, skin_temperature_c=33.0, ambient_temperature_c=24.0,
        acc_band_cardiac=0.01, ibi_intervals_ms=[820, 810], ibi_timestamps_ms=[1000, 1810],
    )
    base.update(kw)
    return AcquisitionBatchMessage(**base)


def test_wrist_batch_declares_core_capabilities():
    frame = frame_from_acquisition_v2(_batch())
    assert frame.has(Capability.IBI)
    assert frame.has(Capability.HR)
    assert frame.has(Capability.MOTION)
    assert frame.has(Capability.SKIN_TEMP)      # valid temp present
    assert not frame.has(Capability.PPG)        # no raw PPG arrays
    assert frame.capabilities.profile == "wrist_ppg"


def test_skin_temp_absent_when_zero():
    frame = frame_from_acquisition_v2(_batch(skin_temperature_c=0.0))
    assert not frame.has(Capability.SKIN_TEMP)
    assert frame.skin_temp_c is None


def test_ppg_capability_only_when_arrays_present():
    frame = frame_from_acquisition_v2(_batch(ppg_green=[1, 2, 3], ppg_timestamps_ms=[1, 2, 3]))
    assert frame.has(Capability.PPG)
    assert frame.ppg is not None
    assert frame.ppg.green == [1, 2, 3]


def test_frame_carries_same_ibi_and_hr():
    b = _batch()
    frame = frame_from_acquisition_v2(b)
    assert frame.ibi.intervals_ms == b.ibi_intervals_ms
    assert frame.hr_bpm == b.heart_rate_bpm
    assert frame.motion_energy == b.motion_energy()
    assert frame.raw is b   # escape hatch preserved for legacy engines


def test_pipeline_ingest_frame_equivalent_to_acquisition():
    """ingest_acquisition (via frame) feeds the IBI buffer identically."""
    from affectus.engine.pipeline import PhysiologyPipeline

    p1 = PhysiologyPipeline()
    p1.ingest_acquisition(_batch())
    # the IBI buffer received both beats
    assert p1.ibi_buffer.entries_in_last_ms(120_000, end_ms=1810)
