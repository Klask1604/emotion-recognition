"""
SensorFrame: the common in-process message every device adapter produces and the
pipeline consumes. Canonical signal slots are None when the corresponding
capability is absent; `raw` carries the original wire message so the parallel
legacy/research engines keep working unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

from affectus.contract.capabilities import Capability, DeviceCapabilities
from affectus.ingestion.messages import (
    AcquisitionBatchMessage,
    IbiBatchMessage,
    PpgBatchMessage,
)


@dataclass(frozen=True)
class SensorFrame:
    timestamp_ms: int
    capabilities: DeviceCapabilities

    # Canonical slots — None when absent (gated by capabilities).
    ibi: IbiBatchMessage | None = None
    hr_bpm: float | None = None
    skin_temp_c: float | None = None
    ambient_temp_c: float | None = None
    motion_energy: float | None = None
    ppg: PpgBatchMessage | None = None

    # Escape hatch for the legacy/research engines (never used by the core decision).
    raw: AcquisitionBatchMessage | None = None

    def has(self, capability: Capability) -> bool:
        return self.capabilities.has(capability)
