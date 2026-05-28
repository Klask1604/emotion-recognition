# Biofizic — System Architecture

Real-time physiological-state classifier for VR. Two repos:
- **Server** (Python, Docker): MQTT compute + InfluxDB + Grafana — `C:\Users\doltu\Desktop\Licenta`
- **Watch** (Kotlin, Wear OS, Galaxy Watch 7): sensor acquisition + UI

---

## 1. Big picture

```
┌─────────────────────────┐         MQTT          ┌──────────────────────────┐
│   GALAXY WATCH 7 (Kotlin)│  ───── (WiFi) ─────▶  │   SERVER (Python, Docker) │
│                          │                       │                          │
│  sensors → IBI pipeline  │  biofizic/acquisition │  compute-engine          │
│  → AcquisitionAssembler  │  /batch (1 Hz, atomic)│  (compute pipeline)      │
│  → MQTT (PublishSched.)  │  biofizic/cmd/calibrate│ mqtt-logger (writes DB) │
│                          │                       │                          │
│  UI (Compose) ◀──────────│  biofizic/state,      │  ──▶ InfluxDB 3 ──▶ Grafana
│  arousal/confidence      │  /calibration/status  │                          │
└─────────────────────────┘  ◀─────────────────── └──────────────────────────┘
```

Two server processes: **compute-engine** (computes the verdict) and **mqtt-logger**
(writes everything to InfluxDB for dashboards). They do not talk directly — both
subscribe to MQTT.

---

## 2. Watch — from sensor to packet

```
Samsung sensors (different cadences!):
  HR/IBI  ~4s bursts ─┐
  PPG     ~25 Hz     ─┤
  ACC     ~25 Hz     ─┼─▶ IbiPipeline ──▶ AcquisitionAssembler ──▶ PublishScheduler ──▶ MQTT
  Gyro    sensor evt ─┤      (cleans       (ATOMIC bundle with        (1 packet / 1s)
  SkinTemp slow      ─┘       beats)         shared ts_anchor)
```

Each sensor arrives at a different rate. The server's HRV math needs every metric
in a packet to refer to the **same time window**. Solution = **atomic packet**:

- **`IbiPipeline.kt`** — receives beats from the SDK, validates them physiologically,
  reconstructs timestamps when the SDK omits per-beat epochs.
- **`AcquisitionAssembler.kt`**:
  1. Computes **`ts_anchor`** = latest known timestamp across IBI/PPG/motion/temp →
     the server aligns its 30s rolling window to that instant.
  2. **Drains** IBI accumulated since the last publish (not a fixed 1s slice — HR
     ships in ~4s bursts, a 1s slice would drop whole batches).
  3. Computes **`acc_band_cardiac`** = acceleration energy in 0.5–4 Hz (FFT over 8s)
     — the part of wrist motion that overlaps the PPG pulse. The only motion scalar sent.
- **`PublishScheduler.kt`** — sends 1 packet per second.
- **`SensorService.kt`** — foreground service; orchestrates everything, handles
  calibration, owns `WatchStateRepository` (UI state).
- **`MqttSession.kt`** — MQTT connection (publish + subscribe to state/calibration).

---

## 3. MQTT topic map

```
WATCH → SERVER:
  biofizic/acquisition/batch    (1 Hz) IBI + acc/gyro stats + temp + ts_anchor + sync diag
  biofizic/cmd/calibrate        (re)calibration command + reported_arousal

SERVER → WATCH / DASHBOARDS:
  biofizic/state                (30s) epoch verdict (retained, QoS1)
  biofizic/state/live           (1 Hz) live verdict
  biofizic/state/windows        (periodic) HRV over 30/60/90s
  biofizic/live                 (1 Hz) ALIGNED stream, everything on ts_anchor
  biofizic/calibration/status   calibration phase (collecting/done)
  biofizic/legacy/{ppg,wesad,valence}  research engines
```

---

## 4. Server — the compute pipeline (the core)

```
biofizic/acquisition/batch
        │
        ▼
┌─────────────────┐  INGESTION: messages.py parses → AcquisitionBatchMessage
│  INGESTION      │             (IBI + sensor stats + ts_anchor)
└────────┬────────┘
         ▼
┌─────────────────┐  RollingIbiBuffer    (keeps 120s of beats)
│  BUFFERS        │  RollingSensorBuffer (motion energy)
└────────┬────────┘
         ▼
┌─────────────────┐  DSP (signal cleaning):
│  DSP            │   • ibi_filter.py        — timestamp-coherent successive diffs
│                 │   • artifact_correction  — flag beats >20% off the LOCAL median,
│                 │                            interpolate (not delete) → artifact_rate
└────────┬────────┘
         ▼
┌─────────────────┐  compute_features/windows.py + hrv_metrics.py, over 30/60/90s:
│  FEATURES       │   RMSSD, SDNN, pNN50, mean_hr (=60000/mean_IBI),
│  (multi-window) │   Baevsky → Kubios Stress Index (=√Baevsky)
└────────┬────────┘
         ▼
┌──────────────────────────── ENGINE (decision) ───────────────────────────┐
│  signal_quality.py ──▶ Q (0..1) + motion_state(still/moving) + artifact   │
│  baseline.py       ──▶ z_hrv, z_hr (vs YOUR robust log-space baseline)    │
│        ▼                                                                   │
│  pipeline.py FUSION:   z_fused = Q·z_hrv + (1-Q)·z_hr                      │
│                        confidence = Q·Q + (1-Q)·0.7 ──▶ dominant_channel   │
│        ▼                                                                   │
│  state_estimator.py KALMAN  (smooths z_fused; bad epochs barely move it)  │
│        ▼                                                                   │
│  decision_gate.py:                                                         │
│     • arousal 1..10 = round(1 + 9·Φ(z_filtered + reported_offset))        │
│     • cusum.py: sustained-stress alert                                    │
│     • arousal_mapper.py: label (Relaxat..Ridicat), Kubios zones           │
└────────────────────────────────┬─────────────────────────────────────────┘
                                  ▼
                    PhysiologyDecision ──▶ MQTT (state/live/windows)
```

### Two heart-rate numbers (often confused)
- **`hr_sdk` / `hr`** — raw Samsung SDK heart rate, measured ON THE WATCH from the
  optical signal. In `acquisition_batch` and `live`.
- **`mean_hr`** — computed ON THE SERVER from the beats: `60000 / mean(IBI)`
  (`hrv_metrics.py`). In `state`, `state_live`, `windows`.
- If beats are clean, `mean_hr ≈ hr_sdk`; a large gap means dropped/corrupted beats.

---

## 5. Engine modules — what + why (the science)

| Module | Role | Method / citation |
|---|---|---|
| `signal_quality.py` | How good the signal is (Q) | `Q=(1/(1+(A/Amax)²))·(1−P(art\|motion))·motion_factor`; still/moving from own motion baseline (median+MAD) with hysteresis (+4σ enter / +2σ exit). TROIKA/Zhang 2015; Task Force ESC 1996 |
| `baseline.py` | YOUR personal reference | RMSSD/SI log-normal → robust z on `ln(x)`: `z=(ln x − median)/(1.4826·MAD)` (Hampel). Locks after 12 epochs, slides over 60. Persisted to disk |
| `state_estimator.py` | Smart smoothing | Scalar Kalman; measurement variance = `BASE/Q` → low-quality epoch = tiny gain = barely moves estimate. Kalman 1960 |
| `cusum.py` | Sustained-stress alert | `S_t=max(0,S_{t-1}+(z−k))`, alert when `S>h` (k≈0.5, h≈4 in σ). Page 1954 |
| `arousal_mapper.py` | z → 1-10 + label | `arousal=round(1+9·Φ(z+offset))`; offset = probit of self-reported state. Kubios zones (population); Cohen's κ |
| `decision_gate.py` | Assemble final verdict | arousal + label + alert |
| `pipeline.py` | Orchestrator | wires it all, emits `PhysiologyDecision` |

### Motion-tolerant fusion (the VR core)
`hrv_weight = Q`. Still & clean → verdict from **HRV** (precise). Moving / dirty →
verdict leans on **HR** (robust). `dominant_channel` ∈ {hrv, hr, blend, none}.
Reported **confidence** is multi-channel (floors near 0.7 via HR in motion), NOT the
HRV-only Q which collapses — so a moving verdict reads ~60-70% "via HR", not ~3%.

---

## 6. Calibration loop (self-report)

```
Press recalibrate on watch
   │  pick: Calm/Normal/Alert/Stressed → reported_arousal ∈ {0.2,0.4,0.6,0.8}
   ▼
biofizic/cmd/calibrate {reported_arousal}
   ▼
server: reset_baseline(reported) → clears old baseline
        publishes phase="collecting"          ◀── watch shows the SPINNER
   │
   │  stay still: collects 12 resting epochs (~12s+)
   ▼
baseline re-locks (is_ready=true)
        publishes phase="done"                ◀── watch stops the spinner
```

`reported_arousal → offset_z = probit(reported)` so at `z=0` (rest) the displayed
arousal equals what you reported, not a fixed 5. Anchors the scale to you.

Robustness: the watch ignores any calibration/status message whose `ts` predates the
recalibrate request (kills the retained "done" replayed on every (re)subscribe).

---

## 7. Research engines (legacy — parallel, NOT production)

Run in parallel only to justify design choices in the thesis:
- **`raw_ppg.py`** — PPG peak detection (Butterworth 0.5-4Hz + find_peaks),
  reconstructs IBI, pulse amplitude (PPA). Shows PPA collapses under motion.
- **`wesad.py`** — RandomForest trained on WESAD (chest ECG). Outputs `p_stress`;
  demonstrates a foreign-dataset model over-flags on wrist PPG (domain shift).
- **`valence.py`** — valence heuristic (documented negative result; not separable
  from arousal).
- **`toggles.py`** — feature flags (currently all ON for dashboards).

---

## 8. Storage + visualization

```
mqtt-logger ──(dedicated writer thread + queue)──▶ InfluxDB 3 Core ──▶ Grafana (13 dashboards)
   │
   listens to all topics, field allowlist per topic,
   anchors raw PPG samples to server-now, heartbeat every 30s
```

The writer thread + queue decouples InfluxDB writes from the MQTT callback so a slow
write never blocks reception (this fixed the recurring data gaps). The heartbeat
shows flow health: `recv / pts written / err / queue / dropped`.

### InfluxDB measurements
| Measurement | Content |
|---|---|
| `biofizic_state` | epoch verdict (30s) |
| `biofizic_state_live` | live verdict (1 Hz) |
| `biofizic_state_windows` | HRV over 30/60/90s |
| `biofizic_acquisition_batch` | raw 1 Hz packet (HR, acc/gyro, temp, sync diag) |
| `biofizic_live` | aligned 1 Hz stream (everything on ts_anchor) |
| `biofizic_all_data_live` | raw PPG samples + IBI |
| `biofizic_legacy_{ppg,wesad,valence}` | research engines |

### Parameter provenance
- **Watch raw (sensors):** `hr`/`hr_sdk`, `ppg_green`/`ppg_ir`, `ibi_ms`, `skin_temp`
- **Watch computed:** `acc_rms/p90/std`, `gyro_*`, `acc_band_cardiac`, `ts_anchor`, `seq`, `anchor_delay_ms`
- **Server computed:** `rmssd`, `sdnn`, `stress_index`, `mean_hr`, `z_si`, `z_hr`,
  `z_si_filtered`, `arousal_10`, `emotion`, `emotion_baseline`, `motion_state`,
  `signal_quality`, `artifact_rate`, `confidence`, `dominant_channel`, `kalman_gain`, `labels_agree`
- **Server legacy/ML:** `p_stress` (WESAD RF), `valence`, `ppa`, `ppa_z`, `n_peaks`, `ibi_recon_mean`

---

## 9. Session report — 2026-05-27, ~21:03-21:08 (gaming at PC, LoL, keyboard typing)

A real-world test of the motion-tolerant fusion: typing = intermittent wrist motion.

**Verdict stability**
| Time | Arousal | HR | RMSSD | Stress idx | Motion | Confidence |
|---|---|---|---|---|---|---|
| 21:03:05 | 5 Moderat | 91 | 67 | 8.7 | still | 58% via hr |
| 21:03:36 | 5 Moderat | 90 | 61 | 8.5 | **moving** | 65% via hr |
| 21:04:37 | 5 Moderat | 89 | 45 | 9.9 | still | 60% via blend |
| 21:05:38 | 5 Moderat | 86 | 51 | 9.6 | still | 79% via hrv |
| 21:06:09 | 4 Echilibrat | 90 | 40 | 7.4 | still | 86% via hrv |
| 21:07:41 | 5 Moderat | 89 | 59 | 12.8 | still | 88% via hrv |

**Channel distribution:** 5× via hrv, 3× via hr, 2× via blend, 1 moving epoch.

**Key findings**
1. **Fusion works in the wild:** when typing degraded HRV, the verdict shifted to the
   HR channel and confidence stayed **58-65%** (never collapsed to ~3%). When still,
   it returned to HRV at **79-88%**. Exactly the intended VR behavior.
2. **No jumping:** arousal held steady at 4-5/10 ("focused gaming, not stressed")
   even though RMSSD swung 39→76→45→57 ms. The Kalman + fusion absorbed the noise —
   the original "arousal topăie" complaint is resolved.
3. **HR 86-91 bpm, stress index 7.4-12.8 (Echilibrat/Moderat):** consistent with
   alert-but-calm gaming.
4. **Infrastructure healthy:** 0 errors, 0 disconnects; logger `recv≈5.7/s, err=0,
   queue=0, dropped=0`.

**Conclusion:** the system behaves correctly under intermittent real motion — it
keeps a confident, stable verdict by leaning on HR when HRV is briefly corrupted,
without freezing or jumping.
