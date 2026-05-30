# biofizic/ingestion/

## Purpose
The boundary layer: turn the raw watch MQTT payload into typed, validated domain
objects so the rest of the pipeline never touches JSON.

## Inputs
- The decoded JSON dict of `biofizic/acquisition/batch` (schema v2), parsed by
  `services/compute_engine.py` (`_parse_acquisition`).
- ① watch raw + ② watch computed fields: `ts_publish`, `ts_anchor`, `seq`,
  `heart_rate_bpm`, acc/gyro stats, `acc_band_cardiac`, skin/ambient temp,
  `ibi_intervals_ms` + `ibi_timestamps_ms`, optional raw PPG arrays.

## Outputs
- `AcquisitionBatchMessage` — the atomic 1 Hz frame, plus helpers:
  - `.to_ibi_batch()` → `IbiBatchMessage` (beats for HRV)
  - `.to_sensor_batch()` → `SensorBatchMessage` (HR/acc/gyro/temp stats)
  - `.motion_energy()` → `acc_band_cardiac` (fallback `acceleration_rms`)
- `InterbeatIntervalEntry` — one beat `(interval_ms, timestamp_ms)`, the atom the
  HRV math consumes.

## Key files
| File | Role |
|---|---|
| `messages.py` | All dataclasses: `AcquisitionBatchMessage`, `IbiBatchMessage`, `SensorBatchMessage`, `PpgBatchMessage`, `InterbeatIntervalEntry` |

## Data flow
```
MQTT JSON ─▶ compute_engine._parse_acquisition ─▶ AcquisitionBatchMessage
                                                      ├─ to_ibi_batch()    ─▶ RollingIbiBuffer (compute_features)
                                                      ├─ to_sensor_batch() ─▶ pipeline (HR, motion)
                                                      └─ motion_energy()   ─▶ signal_quality (engine)
```

## Depends on / Used by
- **Depends on:** stdlib only (pure dataclasses).
- **Used by:** `services/compute_engine.py`, `compute_features/`, `engine/`.
- `ts_anchor` carried here is what keeps every downstream metric on one clock.
