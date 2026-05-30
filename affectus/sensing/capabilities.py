"""
Capability-driven sensor contract.

Any wearable (wrist GW7/GW8, chest strap, head-worn) is described not by its
model but by WHICH SIGNALS it carries. A device adapter (see sensing/adapters/)
maps the raw device payload into a single `SensorFrame`; the pipeline then runs
only the feature-engines whose required capabilities are present in that frame —
no branching on "is this a GW8".

This is the explicit form of two patterns the codebase already used implicitly:
the zero-weight no-op in engine.decision._fuse, and the conditional engine
construction in legacy.__init__. Here, presence is a capability flag (run the
channel at all) and usefulness stays a weight (how much it contributes) — two
clean gates, never one overloaded.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from affectus.ingestion.messages import (
    AcquisitionBatchMessage,
    IbiBatchMessage,
    PpgBatchMessage,
)


class Capability(str, Enum):
    """A signal a wearable can provide. The pipeline asks `frame has X?`."""

    IBI = "ibi"            # inter-beat intervals (from PPG peaks OR ECG R-peaks)
    PPG = "ppg"            # raw optical pulse waveform (green/ir samples)
    MOTION = "motion"      # accelerometer / gyroscope motion energy
    SKIN_TEMP = "temp"     # peripheral skin temperature
    HR = "hr"              # SDK-processed heart-rate scalar
    # Declared-but-unimplemented slots — future adapters/channels plug in here:
    EDA = "eda"            # electrodermal activity (GW8)
    EEG = "eeg"            # electroencephalography (head-worn)
    ECG_RAW = "ecg_raw"    # raw ECG (chest strap)
    RESP = "resp"          # respiration belt


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a frame carries (present) and which threshold profile applies.

    `profile` keys into sensing.profiles.PROFILES (e.g. 'wrist_ppg', 'chest_ecg')
    so the shared algorithms read device-appropriate thresholds without ever
    branching on device identity."""

    present: frozenset[Capability]
    profile: str = "wrist_ppg"

    def has(self, capability: Capability) -> bool:
        return capability in self.present

    def has_all(self, required: frozenset[Capability]) -> bool:
        return required <= self.present


@dataclass(frozen=True)
class SensorFrame:
    """The common contract every adapter produces and the pipeline consumes.

    Canonical signal slots are None when the corresponding capability is absent.
    `raw` is an escape hatch carrying the original wire message so the parallel
    legacy/research engines keep working unchanged.
    """

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
