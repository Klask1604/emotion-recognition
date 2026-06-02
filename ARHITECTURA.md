# Arhitectura sistemului Biofizic - referinta completa

> Sursa unica de adevar. Scris din cod (nu din memorie). Verifica de aici
> inainte de orice afirmatie despre sistem. Ultima verificare: 2026-06-02.

## 1. Fluxul general (cap-coada)

```
Ceas (wrist) --WiFi--> MQTT --> Engine Python (Docker) --> InfluxDB --> Grafana
                                       |                                    
                                       +--> verdict inapoi pe ceas          
                                       +<--> Unity / VR (context + verdict) 
```

Chest si Headset: structura pregatita in cod (DDD), DAR NEIMPLEMENTATE
(folderele de module sunt goale). In diagrame = linie punctata, via BLE -> telefon.

## 2. Ce trimite ceasul (batch, ~1 mesaj/secunda)

Topic IN: `biofizic/acquisition/batch`. Campuri (din `compute_engine.py` _parse):

- **Cardiac:** IBI (ms + ts), PPG verde, PPG IR, HR (bpm)
- **Temperatura:** skin_temp_c, ambient_temp_c
- **Miscare (acc):** acc_rms, acc_p90, acc_std, acc_band_cardiac
- **Miscare (gyro):** gyro_rms, gyro_p90, gyro_std

NU trimitem doar puls+temperatura. Trimitem si accelerometru + giroscop.
Rolul acc/gyro: NU sunt semnal de emotie. Sunt **gate de calitate** - detecteaza
miscarea, elimina ferestrele contaminate, si decid cand suntem "still" (nemiscat)
ca sa se invete baseline-ul DOAR in repaus.

## 3. Timpii / intervale (din config.py)

| Constanta | Valoare | Ce e |
|---|---|---|
| Batch de la ceas | ~1 s | un mesaj pe secunda |
| ANALYSIS_WINDOW_SECONDS | 30, 60, 90 s | ferestre HRV calculate in paralel |
| PRIMARY_DECISION_WINDOW_SECONDS | 60 s | fereastra care da decizia |
| Buffer IBI (HRV_LOOKBACK) | 90 s | buffer glisant |
| EPOCH_PUBLISH_INTERVAL_SECONDS | 30 s | **epoca** = verdict oficial la 30s |
| WINDOWS_PUBLISH_INTERVAL_SECONDS | 5 s | dashboard ferestre |
| Live | ~1 s | stare rapida (biofizic/live) |

**Fereastra vs Epoca (NU confunda):**
- Fereastra = cate date FOLOSESTE (60s, se uita inapoi)
- Epoca = cat de DES da verdict oficial (30s)
- Deci: la fiecare 30s, ia ultimele 60s de date, scoate un verdict.

## 4. Cum se decide AROUSAL (axa care merge)

In `affectus/engine/decision.py`:
1. Din HRV (RMSSD, SDNN) + temperatura pielii.
2. z-score PERSONAL fata de baseline-ul propriu de repaus (nu o medie generala).
3. z > 0 = HRV/temp sub baseline = activare simpatica = arousal ridicat.
4. Mapare prin CDF personal: `personal_arousal_10(z)` -> scara 0-10.
5. Filtrare Kalman + gate de calitate (semnal slab nu misca estimarea).
6. Histerezis pe afisaj (LIVE_AROUSAL_HYSTERESIS_TICKS = 3).

Baseline arousal: se blocheaza dupa ~6 min de epoci de repaus.
Robust cross-subject (merge si pe oameni noi).

## 5. VALENTA - ce am facut

Valenta NU se citeste direct din puls ca arousalul. 4 modele ruleaza in paralel,
toate folosesc engine-ul `valence_wesad_engine.py` (drop-in compatible):

| Model | Antrenat pe (label) | Tip |
|---|---|---|
| WESAD | stres(2) vs amuzament(3) | PROXY (2 conditii, nu valenta reala) |
| CASE | valenta auto-raportata >=7 vs <=3 | valenta REALA (joystick 1-9) |
| EEVR | HVHA vs LVHA (ambele arousal mare) | valenta reala, izolata de arousal |
| Polaritate | neutru/negativ/pozitiv (3 clase) | + CORAL + arousal gate |

**Conversia la valenta (toate 4):** `valence_z = 2 * p_positive - 1` (rescalare
liniara [0,1]->[-1,+1]). p_positive = probabilitatea clasei pozitive data de model.
Deci NU e "valenta din puls" pura - e "cat seamana cu pozitiv" interpretat ca valenta.

**Neutru:** WESAD/CASE/EEVR au doar 2 clase (fara neutral) - intentionat, ca sa
izoleze axa de valenta. Neutralul vine din logica de dupa: gate confidenta (0.7) +
deadband. Polaritatea are clasa neutral proprie (baseline+meditatie).

## 6. VERDICTUL 2D (cadran Russell)

In `affectus/shared/emotion.py`. Combina arousal x valenta:
- coduri: 0 Neutru, 1 Calm, 2 Trist, 3 Bucuros, 4 Stresat
- arousal sus + valenta + -> Bucuros; arousal sus + valenta - -> Stresat
- arousal jos + valenta + -> Calm; arousal jos + valenta - -> Trist
- valenta nesigura / in deadband -> Neutru

**Stabilizare** (`ValenceVerdictStabilizer`, un stabilizator PER model wesad/eevr/case):
- CONF_MIN = 0.7 (sub asta -> Neutru, "nu stiu" onest)
- MEDIAN_WINDOW = 5 epoci (mediana rolling)
- HYSTERESIS_TICKS = 3 (cadranul se schimba doar daca persista 3 epoci)
- publica `emotion_code` gata calculat -> dashboardul DOAR il citeste (zero CASE WHEN in SQL)

## 7. Topicuri MQTT

**IN (engine subscribe):**
- biofizic/acquisition/batch - datele de la ceas
- biofizic/cmd/calibrate - comanda recalibrare
- biofizic/cmd/feedback - eticheta de emotie de la user (cadran Russell)
- biofizic/context - context scena VR (Unity)
- biofizic/ecg/calibrate - calibrare ECG (finger-hold)
- biofizic/hello - handshake capabilities
- biofizic/ppg/ondemand - cerere PPG batch

**OUT (engine publish):**
- biofizic/live - stare live (~1s)
- biofizic/state - verdict epoca (30s, oficial, retinut)
- biofizic/state/live, biofizic/state/windows
- biofizic/legacy/wesad, .../polarity, .../ppg, .../resp, .../feedback
- biofizic/hello/ack, biofizic/calibration/status

## 8. Modele pe disc (models/)

Folosite live: valence_wesad, valence_case, valence_eevr, polarity (.joblib).
Plus: deap_valence_fd, model, model_v3, motion_har_wisdm, wesad_rf (diverse/vechi).

## 9. REZULTATE valenta (validate, reproductibile)

Binar negativ vs pozitiv, balanced accuracy:

| Dataset | GENERAL (strain, LOSO) | PERSONAL (calibrat) |
|---|---|---|
| WESAD | 85% | 94% |
| CASE | 60% | 82% |
| EEVR | 56% | 72% |
| EMOGNITION | 50% | 69% |

- **Constatare centrala:** valenta slaba cross-subject (~50-60%, spre sansa),
  puternica personal (69-94%). Personalizarea ESTE solutia.
- **Intensitate:** precizia creste cu intensitatea (CASE: usor 82% -> intens 87%).
- **Test arousal-deghizat:** valenta REZISTA cand scoti features de arousal
  (cade doar 1% pe WESAD) -> e valenta reala, nu arousal mascat.
- **Ce NU merge (testat, respins):** combinare dataseturi (58%), EEG/FAA (43%),
  fuziune EDA+PPG (56%), CORAL pe valenta cross-subject, ferestre lungi, online learning.

## 10. Reglaje / decizii recente (2026-06-02)

- **reactivity = high** (data/reactivity_profile.json) -> deadband x1.4. Pus ca sa
  evite "happy fals": baseline-ul de valenta s-a calibrat negativ (-0.58, subiect
  constant stresat), deci abateri mici pareau pozitive. Cu deadband mai lat, abaterile
  mici raman Neutru. Radacina reala: baseline calibrat in stare non-neutra -> ar
  trebui recalibrat cand subiectul e odihnit. [[biofizic-valenta-limitare]]
- **Verdict mutat in Python** (din SQL): fiecare model publica emotion_code stabilizat;
  dashboardurile doar citesc (zero CASE WHEN de verdict). Panelul circumplex XY ramane
  cu SQL (doar geometrie de plot, nu verdict - corect).

## 11. Stare implementare

- **wrist:** IMPLEMENTAT si testat (singura parte testata pe subiect real).
- **chest, headset:** structura DDD pregatita, module GOALE, NEIMPLEMENTATE.
- **Unity/VR:** plan context->valenta, partial; bucla feedback nu e gata live.
