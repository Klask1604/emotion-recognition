# biofizic/dsp/

## Purpose
Digital signal processing: clean the raw beat series before any HRV statistic is
computed, and (for the research engine) recover beats from the raw PPG wave.

## Inputs
- A list of `InterbeatIntervalEntry` (from `ingestion`, buffered by `compute_features`).
- For PPG peaks: raw `ppg_green` samples + their timestamps (① watch raw, legacy only).

## Outputs
- **Cleaned IBI series + `artifact_rate`** (fraction of beats corrected) — the standard
  HRV signal-quality index (Task Force ESC 1996).
- **Successive interval differences** computed only across timestamp-coherent beat pairs.
- **Detected PPG peaks + reconstructed IBI + pulse amplitude (PPA)** for the legacy engine.

## Key files
| File | Role |
|---|---|
| `artifact_correction.py` | Flag beats >20% off the LOCAL median (±5 neighbours), **interpolate** (not delete); returns `(entries, artifact_rate)` |
| `ibi_filter.py` | `successive_interval_differences` using only timestamp-coherent pairs (feeds RMSSD) |
| `ppg_peaks.py` | Butterworth 0.5–4 Hz band-pass + `find_peaks` → peaks, PPA, reconstructed IBI (scipy, lazy import) |

## Data flow
```
raw IBI ─▶ artifact_correction (local-median, interpolate) ─▶ clean IBI + artifact_rate
clean IBI ─▶ ibi_filter (coherent successive diffs) ─▶ RMSSD input
raw PPG  ─▶ ppg_peaks (band-pass + find_peaks) ─▶ peaks / PPA / recon IBI  (legacy)
```

## Depends on / Used by
- **Depends on:** `ingestion` (entry types), `config` (physiological band), numpy/scipy.
- **Used by:** `compute_features/hrv_metrics.py` (cleaned beats), `legacy/raw_ppg.py` (peaks).
- Why interpolate, not delete: deleting beats biases RMSSD; local-median (not whole-window)
  avoids flagging genuine HRV as artifacts.
