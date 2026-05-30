# biofizic/legacy/

## Purpose
Parallel **research** engines that run alongside production but never feed the verdict.
They exist to justify design choices in the thesis (comparison dashboards + negative results).

## Inputs
- The same per-epoch data as production: `SensorBatchMessage`/`AcquisitionBatchMessage`,
  the `MultiWindowResult`, and the personal `RestBaselineStore`.
- Gated by `toggles.py` (currently all ON for the dashboards).

## Outputs (published on `biofizic/legacy/*`)
- `raw_ppg.py` → `ppa`, `ppa_z`, `n_peaks`, `sample_rate_hz`, `ibi_recon_mean`.
- `wesad.py` → `p_stress` (RandomForest probability from a chest-ECG-trained model).
- `valence.py` → `valence` [-1,1] + its inputs (`rmssd_z`, `ppa_z`).

## Key files
| File | Role |
|---|---|
| `__init__.py` | `LegacyEngines` facade: `.active`, `.run(batch, result, baseline)` → outputs |
| `raw_ppg.py` | PPG peak detection / PPA / reconstructed IBI (uses `dsp/ppg_peaks`) |
| `wesad.py` | Loads the WESAD model (graceful skip if file missing) → P(stress) |
| `valence.py` | Ad-hoc valence heuristic (documented negative result) |
| `toggles.py` | Feature flags for each engine |

## Data flow
```
batch + result + baseline ─▶ LegacyEngines.run ─┬─▶ raw_ppg  ─▶ biofizic/legacy/ppg
                                                ├─▶ wesad    ─▶ biofizic/legacy/wesad
                                                └─▶ valence  ─▶ biofizic/legacy/valence
```

## Depends on / Used by
- **Depends on:** `dsp`, `compute_features`, `engine/baseline`, the trained model from `train/`.
- **Used by:** `services/compute_engine.py` (`_publish_legacy`), then the comparison dashboards.
- Must never raise into production: model-load is wrapped in try/except (graceful skip).
