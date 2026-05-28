# train/

## Purpose
Offline training of the WESAD-based stress model used by the legacy/research engine.
Run once; produces a model file the live `legacy/wesad.py` engine loads.

## Inputs
- The **WESAD** dataset (chest-ECG + wrist signals; ~17 GB unzipped) under `datasets/`.
- Chest ECG → R-peaks → IBI, fed through the **same** server HRV code
  (`compute_features.compute_hrv_from_entries`) so features match production.

## Outputs
- A trained **RandomForest** classifier (rest vs stress), saved under `models/`
  (mounted into the compute container). Reported: accuracy ≈ 0.836, stress F1 ≈ 0.653.
- Used live to emit `p_stress` on `biofizic/legacy/wesad`.

## Key files
| File | Role |
|---|---|
| `train_wesad.py` | Load WESAD → R-peaks → our HRV features → RandomForest with LOSO cross-validation → save model |

## Data flow
```
WESAD chest ECG ─▶ R-peaks ─▶ IBI ─▶ compute_hrv_from_entries (same as server)
                                              │
                                  RandomForest (LOSO CV) ─▶ models/wesad_*.pkl
                                              │
                              (live) legacy/wesad.py ─▶ p_stress
```

## Depends on / Used by
- **Depends on:** `compute_features` (shared HRV), scikit-learn, the WESAD dataset.
- **Used by:** `biofizic/legacy/wesad.py` at runtime; the Determinist-vs-WESAD dashboard.
- Caveat (thesis): WESAD is chest ECG / E4 wrist; applying it to GW7 wrist PPG is a
  domain shift — expected to be noisier / more false-positive (the point of the comparison).
