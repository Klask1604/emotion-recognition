# Affectus — Arhitectura (capability-driven, straturi DDD)

Clasificator de afect (arousal + valență) din senzori purtabili. Datele de la
orice wearable intră printr-un **contract comun** (SensorFrame), iar pipeline-ul
rulează doar engine-urile pentru care frame-ul are semnalele necesare —
**fără să știe ce model de ceas e** (capability-driven, nu device-hardcoded).

## Principiul de bază (cum citești diagrama)

```
WEARABLE (orice)                SERVER (Python)
  declară ce semnale are  ──▶  ADAPTER ──▶ SensorFrame ──▶ PIPELINE ──▶ DECIZIE
  (PPG? IBI? EDA? temp?)       (per sursă)  (contract     (rulează    (arousal +
                                            comun)         doar ce    valență)
                                                           se poate)
```

Regula de aur:
- **CE calculezi** (HRV, vascular, signal quality) = ALGORITM, scris o dată, partajat.
- **DE UNDE vin datele** (GW7, GW8, chest) = ADAPTER, per sursă.
- **CÂT de strict** (praguri) = PROFIL, per modalitate de senzor.

---

## Harta folderelor

```
/affectus                          PACHETUL PRINCIPAL (clasificatorul)
│
├── config.py                      Toate pragurile numerice + zonele Kubios, fiecare
│                                  adnotat cu sursa (LITERATURE / EMPIRICAL / INFRA).
├── logging.py                     Formatarea blocurilor de log [BASELINE]/[QUALITY]/...
├── _bootstrap.py                  Adaugă rădăcina proiectului pe sys.path (rulare locală).
│
├── /sensing                       ►► STRATUL DE CONTRACT (DDD: infrastructure/boundary)
│   │                              Aici intră orice wearable. Singurul loc cu cod
│   │                              specific-device.
│   ├── capabilities.py            Capability (enum: IBI/PPG/MOTION/SKIN_TEMP/HR/
│   │                              EDA/EEG/...), DeviceCapabilities (ce are device-ul +
│   │                              numele profilului), SensorFrame (CONTRACTUL COMUN
│   │                              pe care îl consumă tot pipeline-ul).
│   ├── profiles.py                SensorProfile per modalitate: praguri device-dependente
│   │                              (wrist_ppg artifact_max=0.15; chest_ecg=0.05 slot viitor).
│   └── /adapters                  Mapează payload-ul brut al fiecărui device → SensorFrame.
│       ├── __init__.py            Registry: schema → adapter (ACQUISITION_ADAPTERS).
│       └── schema_v2.py           Adapterul Galaxy Watch: AcquisitionBatchMessage → SensorFrame.
│                                  (chest_ecg.py, eda.py, eeg.py = sloturi viitoare)
│
├── /ingestion                     ►► DTO de pe sârmă (formatul brut MQTT)
│   └── messages.py                AcquisitionBatchMessage (frame-ul atomic v2 de la ceas),
│                                  IbiBatchMessage, SensorBatchMessage, PpgBatchMessage,
│                                  InterbeatIntervalEntry. + helperi to_ibi_batch() etc.
│
├── /dsp                           ►► PROCESARE SEMNAL (DDD: domain — algoritmi puri)
│   ├── ibi_filter.py              Filtru fiziologic IBI (300-2000ms) + diferențe succesive
│   │                              cu verificare de coerență temporală (nu peste goluri).
│   ├── artifact_correction.py     Corecție artefacte IBI prin interpolare (nu ștergere),
│   │                              local-median; întoarce artifact_rate.
│   └── ppg_peaks.py               Detecție vârfuri PPG (band-pass + find_peaks), PPA.
│
├── /compute_features              ►► METRICI HRV (DDD: domain — partajat, agnostic device)
│   ├── hrv_metrics.py             RMSSD/SDNN/pNN50/Baevsky SI/Kubios SI din IBI.
│   ├── windows.py                 Ferestre glisante 30/60/90s + RollingIbiBuffer /
│   │                              RollingSensorBuffer (retenție pe timp).
│   └── results.py                 Dataclasses rezultat: HrvMetrics, MultiWindowResult,
│                                  PhysiologyDecision (forma servită ceasului + dashboard).
│
├── /engine                        ►► NUCLEUL DECIZIEI (DDD: domain core + application)
│   ├── pipeline.py                ORCHESTRATORUL. ingest_frame() rutează slot-urile
│   │                              frame-ului în buffere; run() = buffere → HRV multi-
│   │                              fereastră → signal quality → baseline → decizie.
│   ├── decision.py                decide(): fuziune multi-canal (_fuse = medie ponderată),
│   │                              Kalman, CUSUM alertă, mapare arousal 1-10. FusionChannel.
│   ├── registry.py                ►► REGISTRY-UL CAPABILITY-GATED. ChannelSpec(name,
│   │                              requires, evaluate) + CHANNEL_REGISTRY. build_channels()
│   │                              rulează un canal DOAR dacă frame-ul are capabilitățile lui.
│   ├── signal_quality.py          Gate de calitate (deterministic): motion energy + artifact
│   │                              rate → Q ∈ [0,1] + still/moving. Citește pragul din PROFIL.
│   ├── baseline.py                RestBaselineStore: baseline personal robust (median+MAD
│   │                              log-space) → z-score personal. Calibrare ~24s.
│   ├── arousal_mapper.py          z-score → arousal 1-10 (CDF personal) sau zonă Kubios.
│   └── /channels                  ►► CANALELE DE FUZIUNE (fiecare = un semnal de afect)
│       ├── temperature.py         Canal temperatură (vasoconstricție → arousal). ȘABLONUL.
│       ├── respiration_rsa.py     Respirație din RSA (IBI) — research.
│       ├── respiration_ppg.py     Respirație din amplitudine PPG — research.
│       └── valence_wesad.py       ►► CANAL VALENȚĂ (modelul WESAD 84.5%). Primul model ML
│                                  în decizie. Plafonat, dezactivat default (cap=0).
│
└── /legacy                        ►► ENGINE-URI RESEARCH (paralele, NU în decizia VR)
    ├── __init__.py                LegacyEngines: construiește condiționat engine-urile
    │                              activate prin toggles; publică pe biofizic/legacy/*.
    ├── toggles.py                 Flag-uri ON/OFF per engine de research.
    ├── valence_features.py        ►► EXTRACTORUL UNIFICAT de valență (vascular+FD+morph+HRV).
    │                              Folosit ȘI la antrenare ȘI în canal (train/serve parity).
    ├── ppg_vascular_features.py   Features ton vascular (perfusion index) — cel mai bun
    │                              predictor de valență (mecanism determinist).
    ├── ppg_hrv_features.py        Features HRV (RMSSD/pNN50/LF-HF din IBI).
    ├── valence_ppg_fd.py          Features frequency-domain (armonice puls).
    ├── valence_ppg_morph.py       Features morfologice (forma undei).
    ├── valence_fd_engine.py       Engine research care publică features FD live.
    ├── respiration_compare.py     Comparator RSA vs PPG-amplitude.
    ├── valence.py                 Heuristică valență veche (rezultat negativ, dezactivat).
    ├── wesad.py                   Model stres WESAD (rezultat negativ domain-shift).
    └── raw_ppg.py                 Parsare PPG brut + PPA.
```

---

## Straturile DDD (cum se mapează folderele)

| Strat DDD | Folder(e) | Ce conține |
|---|---|---|
| **Infrastructure / Boundary** | `sensing/`, `ingestion/` | Contractul + adaptere + DTO de pe sârmă. Singurul loc cu cod device-specific. |
| **Domain (algoritmi puri)** | `dsp/`, `compute_features/` | HRV, filtre, ferestre. Agnostic la device, scris o dată. |
| **Domain core** | `engine/decision.py`, `signal_quality.py`, `baseline.py`, `channels/` | Logica de afect: fuziune, calitate, baseline, canale. |
| **Application (orchestrare)** | `engine/pipeline.py`, `engine/registry.py` | Leagă totul: rutează frame-ul, dispecerizează canalele după capabilități. |
| **Research (separat)** | `legacy/` | Engine-uri paralele de cercetare, nu ating decizia de producție. |

---

## Fluxul unei decizii (end-to-end)

```
1. Ceasul publică MQTT biofizic/acquisition/batch (schema v2)
2. services/compute_engine.py::_parse_acquisition → AcquisitionBatchMessage (ingestion)
3. sensing/adapters/schema_v2.py → SensorFrame (declară: IBI, HR, MOTION, +SKIN_TEMP/PPG)
4. engine/pipeline.py::ingest_frame → rutează slot-urile în buffere (doar ce există)
5. pipeline.run() → HRV multi-fereastră (compute_features) → signal_quality (cu profilul)
6. engine/registry.py::build_channels → construiește DOAR canalele cu capabilitățile prezente
   (wrist: hrv + hr + temp; valence_wesad e skip fiindcă e dezactivat)
7. engine/decision.py::decide → _fuse(canale) → Kalman → CUSUM → arousal 1-10
8. PhysiologyDecision → MQTT biofizic/state (ceasul + Unity + Grafana)
```

## De ce e extensibil (adăugarea unui wearable nou)

GW8 cu EDA, sau un chest strap, NU cere modificări în algoritmi. Doar:
1. Un **adapter** nou în `sensing/adapters/` (mapează semnalul brut → SensorFrame, declară EDA).
2. Un **profil** nou în `sensing/profiles.py` (pragurile lui).
3. Un **canal** nou în `engine/channels/` + înregistrat în `registry.py` cu `requires={EDA}`.

Pipeline-ul îl activează automat când un frame declară EDA. Nicăieri nu apare „dacă e GW8".
