"""
Capability contract: a wearable is described by WHICH SIGNALS it carries, not by
its model. The device announces its capabilities at the handshake
(contract/handshake.py); the pipeline then runs only the fusion channels whose
required capability was declared — no branching on "is this a GW8".

Presence is a capability flag (run the channel at all); usefulness stays a weight
(how much it contributes). Two clean gates, never one overloaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class Capability(str, Enum):
    """A signal a wearable can provide. The pipeline asks `frame has X?`."""

    IBI = "ibi"            # inter-beat intervals (from PPG peaks OR ECG R-peaks)
    PPG = "ppg"            # raw optical pulse waveform (green/ir samples)
    MOTION = "motion"      # accelerometer / gyroscope motion energy
    SKIN_TEMP = "temp"     # peripheral skin temperature
    HR = "hr"              # SDK-processed heart-rate scalar
    # Declared-but-unimplemented slots — future adapters/modules plug in here:
    EDA = "eda"            # electrodermal activity (GW8)
    EEG = "eeg"            # electroencephalography (head-worn)
    ECG_RAW = "ecg_raw"    # raw ECG (chest strap)
    RESP = "resp"          # respiration belt


@dataclass(frozen=True)
class DeviceCapabilities:
    """What a frame carries (present) and which threshold profile applies.

    `profile` keys into a device family's profile (e.g. 'wrist_ppg', 'chest_ecg')
    so the shared algorithms read device-appropriate thresholds without ever
    branching on device identity."""

    present: frozenset[Capability]
    profile: str = "wrist_ppg"

    def has(self, capability: Capability) -> bool:
        return capability in self.present

    def has_all(self, required: frozenset[Capability]) -> bool:
        return required <= self.present
