"""Sensor adapters: map a device's raw payload to the common SensorFrame.

Registry of wire-schema -> adapter. The schema-v2 wrist adapter is the only one
implemented; chest_ecg / eda / eeg are future slots that will register here.
"""

from __future__ import annotations

from affectus.sensing.adapters.schema_v2 import frame_from_acquisition_v2

# schema version -> adapter callable (AcquisitionBatchMessage -> SensorFrame)
ACQUISITION_ADAPTERS = {
    2: frame_from_acquisition_v2,
}
