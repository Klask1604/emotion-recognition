# Biofizic — Architecture (master / high-level)

Real-time physiological-state classifier for VR. A **Galaxy Watch 7** streams sensor
data over MQTT to a **Python server** (Docker) that turns it into a continuous arousal
verdict, stores everything in **InfluxDB 3**, and visualizes it in **Grafana**. The
watch also renders the live verdict.

This file defines the system top-down. Each big folder has its own `ARCHITECTURE.md`
with exact **input → output**; this master links them and shows how they connect.
For the narrative explanation + real session reports, see [`docs/architecture.md`](docs/architecture.md).

---

## Big picture

```
┌─────────────────────────┐         MQTT          ┌──────────────────────────┐
│   GALAXY WATCH 7 (Kotlin)│  ───── (WiFi) ─────▶  │   SERVER (Python, Docker) │
│  sensors → IBI pipeline  │  biofizic/acquisition │  compute-engine           │
│  → AcquisitionAssembler  │  /batch (1 Hz, atomic)│  (compute pipeline)       │
│  → MQTT                  │  biofizic/cmd/calibrate│ mqtt-logger (writes DB)  │
│  UI (Compose) ◀──────────│  biofizic/state, ...  │  ──▶ InfluxDB 3 ──▶ Grafana
└─────────────────────────┘  ◀─────────────────── └──────────────────────────┘
```

Two server processes, both MQTT clients (they never call each other):
- **compute-engine** — computes the verdict.
- **mqtt-logger** — writes every topic to InfluxDB for the dashboards.

---

## End-to-end data flow

```
WATCH                                   SERVER (compute-engine)
sensors ─▶ signal/ (clean beats)        ingestion/ (parse JSON → typed msgs)
        ─▶ acquisition/ (atomic batch)         │
        ─▶ MQTT acquisition/batch ───────────▶ dsp/ (clean IBI, peaks) ─▶ artifact_rate
                                               │
                                        compute_features/ (RMSSD, SI, mean_hr per 30/60/90s)
                                               │
                                        engine/ (quality → baseline z → fusion → Kalman → gate)
                                               │
                                        PhysiologyDecision ─▶ MQTT state/live/windows/live
                                                                   │
WATCH UI ◀── MQTT state/calibration ◀──────────────────────────────┘
                                        mqtt-logger ─▶ InfluxDB ─▶ Grafana
```

The chain is contractual: `ingestion.OUT = dsp.IN`, `dsp.OUT = compute_features.IN`,
`compute_features.OUT = engine.IN`, `engine.OUT = services publish`.

---

## MQTT topic map

```
WATCH → SERVER:
  biofizic/acquisition/batch    (1 Hz) IBI + acc/gyro stats + temp + ts_anchor + sync diag
  biofizic/cmd/calibrate        (re)calibration command + reported_arousal

SERVER → WATCH / DASHBOARDS:
  biofizic/state                (30s) epoch verdict (retained, QoS1)
  biofizic/state/live           (1 Hz) live verdict
  biofizic/state/windows        HRV over 30/60/90s
  biofizic/live                 (1 Hz) ALIGNED stream, everything on ts_anchor
  biofizic/calibration/status   calibration phase (collecting/done)
  biofizic/legacy/{ppg,wesad,valence}  research engines
```

---

## Folder index

### Server (`C:\Users\doltu\Desktop\Licenta`)
| Folder | Purpose | Doc |
|---|---|---|
| `biofizic/` (root) | Central config + log format + bootstrap | [biofizic/ARCHITECTURE.md](biofizic/ARCHITECTURE.md) |
| `biofizic/ingestion/` | Raw MQTT JSON → typed message objects | [ingestion](biofizic/ingestion/ARCHITECTURE.md) |
| `biofizic/dsp/` | Clean the IBI series + PPG peaks | [dsp](biofizic/dsp/ARCHITECTURE.md) |
| `biofizic/compute_features/` | IBI → HRV metrics over 30/60/90s | [compute_features](biofizic/compute_features/ARCHITECTURE.md) |
| `biofizic/engine/` | Features → the verdict (PhysiologyDecision) | [engine](biofizic/engine/ARCHITECTURE.md) |
| `biofizic/legacy/` | Parallel research engines (not production) | [legacy](biofizic/legacy/ARCHITECTURE.md) |
| `services/` | MQTT compute service + InfluxDB logger | [services](services/ARCHITECTURE.md) |
| `scripts/` | Generate dashboards + WESAD comparison | [scripts](scripts/ARCHITECTURE.md) |
| `train/` | Train the WESAD RandomForest model | [train](train/ARCHITECTURE.md) |

### Watch (`...\AndroidStudioProjects\biofizic\app\src\main\java\com\doltu\biofizic`)
| Folder | Purpose | Doc |
|---|---|---|
| `signal/` | Validate/clean SDK beats into IBI window entries | `signal/ARCHITECTURE.md` |
| `acquisition/` | Bundle the atomic 1 Hz acquisition batch | `acquisition/ARCHITECTURE.md` |
| `presentation/` | Service lifecycle, MQTT, Compose UI | `presentation/ARCHITECTURE.md` |

---

## Parameter provenance (glossary)

Three origins, referenced as ① ② ③ throughout the per-folder docs:

- **① Watch raw (sensors):** `hr`/`hr_sdk`, `ppg_green`/`ppg_ir`, `ibi_ms`, `skin_temp`.
- **② Watch computed:** `acc_rms/p90/std`, `gyro_*`, `acc_band_cardiac`, `ts_anchor`, `seq`, `anchor_delay_ms`.
- **③ Server computed:** `rmssd`, `sdnn`, `stress_index`, `mean_hr` (=60000/mean_IBI),
  `z_si`, `z_hr`, `z_si_filtered`, `arousal_10`, `emotion`, `emotion_baseline`,
  `motion_state`, `signal_quality`, `artifact_rate`, `confidence`, `dominant_channel`,
  `kalman_gain`, `labels_agree`.
- **③b Server legacy/ML:** `p_stress` (WESAD), `valence`, `ppa`, `n_peaks`, `ibi_recon_mean`.

`hr_sdk` (①, raw on watch) vs `mean_hr` (③, derived from beats on server) — comparing
them validates beat integrity.
