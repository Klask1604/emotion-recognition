# biofizic/engine/

## Purpose
The decision core: turn HRV features + motion into a single, smoothed, personally-anchored
arousal verdict that stays usable during motion (VR).

## Inputs
- `HrvMetrics` / `MultiWindowHrvResult` (from `compute_features`).
- `SensorBatchMessage` (HR from SDK, motion energy) + `artifact_rate`.
- Optional `reported_arousal` from `biofizic/cmd/calibrate`.

## Outputs
- `PhysiologyDecision`: `display_arousal_10`, labels (Kubios + personal), `stress_index`,
  `z_si`/`z_hr`/`z_si_filtered`, `hrv_weight`, `decision_confidence`, `dominant_channel`,
  `motion_state`, `signal_quality`, `artifact_rate`, `kalman_gain`, `alert`.

## Key files (sub-flow order)
| File | Role |
|---|---|
| `signal_quality.py` | `Q` (0..1) + `motion_state` (still/moving, MAD+hysteresis) + artifact term |
| `baseline.py` | Robust log-space personal baseline → `z_hrv`, `z_hr`; `arousal_offset_z` from self-report; persists to `data/rest_baseline.json` |
| `pipeline.py` | Orchestrator + **fusion**: `z_fused = Q·z_hrv + (1-Q)·z_hr`, `confidence`, `dominant_channel` |
| `state_estimator.py` | Scalar **Kalman**; measurement variance = BASE/Q (bad epoch → tiny gain) |
| `cusum.py` | One-sided CUSUM sustained-stress alert (Page 1954) |
| `decision_gate.py` | Maps `z_filtered`→arousal, runs CUSUM, picks label |
| `arousal_mapper.py` | `arousal = round(1+9·Φ(z+offset))`, Kubios zones, Cohen's κ |

## Data flow
```
features + sensor
   ├─▶ signal_quality ─▶ Q, motion_state, artifact
   ├─▶ baseline ─▶ z_hrv, z_hr (+ offset_z)
   ▼
pipeline: z_fused = Q·z_hrv + (1-Q)·z_hr ; confidence ; dominant_channel
   ▼
state_estimator (Kalman) ─▶ z_filtered
   ▼
decision_gate (+cusum, +arousal_mapper) ─▶ PhysiologyDecision
```

## Depends on / Used by
- **Depends on:** `compute_features`, `ingestion`, `config`, `logging`.
- **Used by:** `services/compute_engine.py` (publishes the decision).
- VR key: motion is a **fusion weight**, never a veto — see [`../../docs/architecture.md`](../../docs/architecture.md) §5.
