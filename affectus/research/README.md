# `affectus/research/` — motoarele de cercetare (paralele cu producția)

## De ce există acest folder

Codul de aici **rulează alături de pipeline-ul de producție, dar nu intră niciodată în
verdictul de arousal** trimis ceasului (`biofizic/state`). Fiecare motor publică pe
`biofizic/legacy/*` și alimentează doar dashboard-urile Grafana, pentru comparație și pentru
teză (inclusiv rezultate negative documentate).

Separarea e o garanție de design, nu doar organizatorică:

- `affectus/engine/`, `affectus/dsp/`, `affectus/contract/`, `affectus/io/` **nu importă
  niciodată** din `affectus/research/`. Calea de producție (arousal) e deterministă, rapidă
  și fără dependențe de cercetare (scipy / scikit-learn).
- Motoarele de aici sunt gated de `toggles.py` și își importă dependențele grele **lazy**, deci
  cu toate toggle-urile pe OFF pachetul se importă fără scipy/sklearn și costă zero CPU.

**Regulă de commit:** `toggles.py` se comite cu research-ul pe valorile de commit (vezi
fișierul), nu cu flagurile pe care le ții ON local pentru o sesiune de colectare.

**Important — axa de valență.** Pe ceas se afișează în prezent **doar arousal-ul** (singura
axă validată). Valența rămâne aici, pe research, **intenționat**: o testăm înainte de a o
promova în verdict. Verdictul de valență (cadran Russell) **trebuie să ruleze** pentru că
alimentează Grafana — de aceea motorul WESAD de valență e cel care produce
`biofizic/legacy/valence_wesad` + verdictul 2D pentru dashboard.

## Structura

| Subpachet | Conținut | Topic MQTT |
|---|---|---|
| `valence/` | features (33), fd_engine, wesad_engine, ppg_fd, ppg_morph, ppg_hrv_features, ppg_vascular_features, wesad_channel, tracks (per-model calibrare) | `biofizic/legacy/valence_*` |
| `respiration/` | compare (RSA vs PPG), estimatorii rsa/ppg | `biofizic/legacy/resp` |
| `ppg/` | raw_ppg (buffer + peaks + IBI reconstruit) | `biofizic/legacy/ppg` |
| `toggles.py` | flagurile build-time pentru toate de mai sus | — |
| `__init__.py` | `ResearchEngines` — facada lazy care rulează doar motoarele toggle-ate | — |

## Ce face fiecare motor și de ce e aici (rezultatul lui)

- **valence/wesad_engine + wesad_channel** — model WESAD (stress vs amusement) pe BVP de
  încheietură, ~85% LOSO pe dataset, dar cu **domain-shift cross-device** către ceas. Rulează
  observed-only ca să judecăm transferul *înainte* de a-l avea încredere în verdict. Cel care
  alimentează verdictul de valență pentru Grafana.
- **valence/eevr_engine (eevr)** — model EEVR pe aceleași 33 de features. Rezultat: **≈ șansă
  (56% LOSO)**. Ținut pentru a arăta divergența pe dashboard.
- **valence/case (case)** — model CASE. Rezultat: **≈ 58%**, singurul semnal real de valență
  dintre cele trei. Comparat live pe semnalul propriu al ceasului.
- **valence/fd_engine + ppg_fd** — 9 features de domeniu-frecvență din armonicele PPG
  (replicare Frontiers 2025). Features-only, fără verdict; pentru teză + training viitor.
- **respiration/compare** — comparator RSA-din-IBI vs respirație-din-amplitudine-PPG, side by
  side. Decide care sursă (dacă vreuna) e suficient de fiabilă pe încheietură pentru a fi fuzată
  ulterior. Research-only.
- **ppg/raw_ppg** — buffer PPG pentru detecția de peak-uri, PPA z-scoring și IBI reconstruit din
  PPG; sursă pentru dashboard-ul de validare (Samsung IBI vs IBI-din-PPG) și infrastructură
  pentru motoarele de valență/respirație.
