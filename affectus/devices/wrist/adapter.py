"""
Adapter: Galaxy Watch acquisition/batch schema v2 -> SensorFrame.

This wraps the existing v2 wire DTO (AcquisitionBatchMessage, still parsed by
services.compute_engine._parse_acquisition) into the device-agnostic SensorFrame
contract. GW7 is unchanged on the wire; only the in-process representation gains
explicit capabilities. Reuses the message's own to_ibi_batch() / motion_energy()
helpers — no new mapping math.
"""

from __future__ import annotations

from affectus.ingestion.messages import AcquisitionBatchMessage, PpgBatchMessage
from affectus.contract.capabilities import Capability, DeviceCapabilities
from affectus.contract.frame import SensorFrame


def frame_from_acquisition_v2(batch: AcquisitionBatchMessage) -> SensorFrame:
    """Build a SensorFrame from a parsed schema-v2 watch batch.

    Capabilities are declared from what the payload actually carries, not from a
    device model: a wrist watch always sends IBI + HR + MOTION; SKIN_TEMP only
    when a valid reading is present; PPG only when the raw arrays were included
    (the on-watch PUBLISH_RAW_PPG toggle)."""
    present = {Capability.IBI, Capability.HR, Capability.MOTION}

    skin_temp = batch.skin_temperature_c if batch.skin_temperature_c > 0 else None
    if skin_temp is not None:
        present.add(Capability.SKIN_TEMP)

    ppg = None
    if batch.ppg_green:
        present.add(Capability.PPG)
        ppg = PpgBatchMessage(
            timestamp_ms=batch.timestamp_anchor_ms,
            green=batch.ppg_green,
            infrared=batch.ppg_infrared,
            sample_timestamps_ms=batch.ppg_timestamps_ms,
        )

    return SensorFrame(
        timestamp_ms=batch.timestamp_anchor_ms,
        capabilities=DeviceCapabilities(frozenset(present), profile="wrist_ppg"),
        ibi=batch.to_ibi_batch(),
        hr_bpm=batch.heart_rate_bpm,
        skin_temp_c=skin_temp,
        ambient_temp_c=batch.ambient_temperature_c or None,
        motion_energy=batch.motion_energy(),
        ppg=ppg,
        raw=batch,
    )
