# biofizic/compute_features/

## Purpose
Compute the HRV feature set the verdict is built on, over several window lengths in
parallel, from a rolling buffer of beats.

## Inputs
- `IbiBatchMessage` ingested into the rolling buffer (from `ingestion`).
- `SensorBatchMessage` motion energy into the sensor buffer.
- An `end_timestamp_ms` (the batch `ts_anchor`) so each window looks back from the same instant.

## Outputs
- `HrvMetrics` per window: `rmssd_ms`, `sdnn_ms`, `pnn50_percent`, `mean_heart_rate_bpm`
  (= 60000 / mean_IBI), `mean_interbeat_interval_ms`, Baevsky raw + **Kubios Stress Index**
  (= √Baevsky), `beat_count`, `covered_seconds`, `artifact_rate`, `is_valid`.
- `MultiWindowHrvResult` (15/30/60/90s) — w30 is the decision window; others diagnostic.
- Result/value types (`results.py`): `PhysiologyDecision`, `WindowResult`, `MultiWindowResult`.

## Key files
| File | Role |
|---|---|
| `windows.py` | `MultiWindowProcessor` (runs HRV per window), `RollingIbiBuffer` (120s retention), `RollingSensorBuffer` (motion) |
| `hrv_metrics.py` | `compute_hrv_from_entries`: RMSSD/SDNN/pNN50/mean_hr + `compute_baevsky_indices` → Kubios SI |
| `results.py` | Dataclasses: `HrvMetrics`, `MultiWindowHrvResult`, `PhysiologyDecision`, `WindowResult`, `MultiWindowResult` |

## Data flow
```
IbiBatchMessage ─▶ RollingIbiBuffer ─(entries in last N s)─▶ MultiWindowProcessor
                                                                  │ per 30/60/90s
                                                                  ▼
                                          compute_hrv_from_entries ─▶ HrvMetrics ─▶ MultiWindowHrvResult
```

## Depends on / Used by
- **Depends on:** `dsp` (clean beats), `ingestion` (entry types), `config` (window lengths), numpy.
- **Used by:** `engine/pipeline.py` (consumes `HrvMetrics` / `MultiWindowHrvResult`).
- `PhysiologyDecision` (defined here) is the contract the `engine` fills and `services` publishes.
