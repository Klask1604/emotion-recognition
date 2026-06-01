# Valence Estimation from Wrist PPG — Technical Report for External Critique

> **Purpose of this document.** This is an honest, self-critical account of an
> attempt to add a *valence* axis (pleasant↔unpleasant) to a real-time affect
> classifier that already estimates *arousal* reliably from a Galaxy Watch. We
> are asking an external reviewer to critique the methodology, the conclusions,
> and whether the approach is salvageable or fundamentally limited. **Be harsh.**

---

## 1. System context

- **Hardware:** Samsung Galaxy Watch 7 (wrist, consumer optical PPG, plus an
  on-demand single-lead ECG that requires the user to hold a finger on the side
  button — Lead-I).
- **Pipeline:** Watch → MQTT → Python compute engine → InfluxDB → Grafana, plus a
  Unity VR front-end. The product is a real-time physiological affect classifier
  for a VR narrative (a bachelor thesis).
- **Arousal (works, validated):** estimated from HRV (RMSSD, Kubios stress index)
  + heart rate, fused by signal quality, smoothed with a scalar Kalman filter,
  mapped to a 1–10 scale via the subject's personal resting baseline (median+MAD
  z-score). This has a **validated causal mechanism** (sympathetic activation →
  HRV↓/HR↑) and behaves sensibly live. **This part is not in question.**
- **Valence (the subject of this report):** an attempt to read the
  pleasant↔unpleasant axis from the same wrist PPG. This is where we are stuck.

The two axes form a Russell (1980) circumplex: emotion = quadrant of
(arousal, valence). Arousal alone cannot separate, e.g., "excited/happy" from
"fear/anger" — both are high-arousal; they differ only in valence. That is the
motivation for wanting valence at all.

---

## 2. The valence model (what we built)

### 2.1 Training data and target
- **Dataset:** WESAD (15 subjects, wrist Empatica E4 **BVP at 64 Hz**).
- **Target = the valence axis isolated from arousal:** binary classification of
  **stress (label 2, negative valence)** vs **amusement (label 3, positive
  valence)**. Both are *high-arousal* states, so a classifier separating them is
  forced to use valence-bearing signal, not arousal. (We deliberately did NOT use
  stress-vs-baseline, which would conflate arousal.)
- **Windowing:** 20 s windows, 10 s step, dominant-label majority vote.

### 2.2 Features (33 total), all from the PPG/BVP waveform
- **Vascular (6):** perfusion_index (AC/DC), ac_amp, dc_level, rise_time,
  pulse_width, reflection_idx. *Mechanistic rationale:* autonomic valence drives
  peripheral vasomotor tone (vasoconstriction in unpleasant/fear, vasodilation in
  some pleasant/anger states; Frontiers Physiol. 2021). This was the single
  strongest family.
- **Frequency-domain (6):** normalised HR-harmonic band powers + ratios
  (replication of Frontiers Physiol. 2025, PMC11893849).
- **Morphology (14):** per-pulse shape — slopes, areas, rise/fall asymmetry
  (same 2025 paper, Table 4).
- **HRV (7):** RMSSD, SDNN, pNN50, LF, HF, LF/HF, mean HR from PPG-derived IBI.

### 2.3 Classifier and validation
- **Model:** RBF-kernel SVM with `probability=True`, `class_weight="balanced"`,
  on z-scored features.
- **Validation:** **Leave-One-Subject-Out (LOSO)** with **balanced accuracy**
  (the honest cross-subject metric; the dataset is imbalanced ~64% negative).
- **Result on WESAD: ~82–85% LOSO balanced accuracy** (vascular features alone
  reached ~85%; adding HRV/morph/FD did not consistently help and sometimes hurt).

### 2.4 Cross-dataset reality check (the important part)
We tested the *same method* on four more datasets to see whether valence-from-PPG
generalises or whether WESAD is a best case:

| Dataset | Sensor | Induction strength | Valence LOSO bal. acc. |
|---|---|---|---|
| WESAD | E4 BVP 64 Hz | strong (stress vs amusement, both high-arousal) | **~82–85%** |
| CASE (clear labels ≤3/≥7) | BVP 1000→100 Hz | continuous joystick, strong | **~60–63%** |
| **EEVR (VR), arousal-matched** | **E4 BVP 64 Hz** | **VR 360 clips (HVHA vs LVHA)** | **56.0%** |
| **EEVR (VR), arousal-mixed** | E4 BVP 64 Hz | VR 360 clips (all-HV vs all-LV) | 54.8% |
| EmoWear | E4 BVP 64 Hz | film clips + retrospective self-report | **~51% (chance)** |
| WPED | wrist PPG 176 Hz | weak (HR identical across emotions) | **~55%** |
| DEAP-Kaggle | — | — | **discarded** (corrupted labels; even EEG gave 47.7% LOSO → confirms the 81% notebooks use inflated leave-one-*point*-out, not LOSO) |

> **EEVR is the decisive negative result.** It is the *same sensor as WESAD*
> (Empatica E4 BVP, 64 Hz), it matches the product's exact context (VR-induced
> emotion, 37 subjects, NeurIPS 2024), and its labels encode the full Russell
> quadrant so we could apply the **identical arousal-isolation trick** that gives
> WESAD 82%: compare HVHA vs LVHA (both high-arousal, differing only in valence).
> It produced **56.0%** — barely above chance — and crucially **only +1.2 pp above
> the arousal-MIXED comparison (54.8%)**. On WESAD the same trick lifted accuracy
> by ~25 pp. The conclusion is sharp: the WESAD result is **not** a transferable
> "method", it is **specific to the extremity of the TSST stressor**. With the same
> hardware and the same arousal-control, ecological VR induction yields no
> cross-subject valence signal in wrist PPG. Adding EEVR's EDA channel changed
> nothing (PPG 55.0% → PPG+EDA 56.2%, +1.2 pp; EDA is arousal-driven, useless on
> an arousal-matched contrast — and the product watch has no EDA anyway).

**Honest conclusion from the cross-dataset sweep:** valence detectability from PPG
**scales with the strength of the affective induction**, not with the method. It
is real at extreme contrast (WESAD) and clear deliberate annotation (CASE), and
collapses to chance on weak/film-clip/self-report datasets. We frame this as a
*graded* limitation, not a binary one, and treat the inflated-validation critique
(leave-one-point-out vs LOSO) as a methodological contribution.

---

## 3. Deploying it live on the watch (where it broke)

### 3.1 Cross-device domain shift (first failure, then fixed)
The model was trained on Empatica E4 BVP (**AC-coupled**, mean ≈ 0). Galaxy Watch
PPG is **DC-coupled** (mean ≈ 65000). Feeding raw watch PPG made `dc_level` land
at z-score ≈ **39718** (should be ~±3); the SVM saturated and output a **constant**
prediction (-0.4694) regardless of input. We confirmed this live.

**Fix:** scale-invariant per-window normalisation in the *shared* feature
extractor (median-centre + MAD-scale) applied identically at train and serve.
After this:
- LOSO on WESAD: 85% → **82.4%** (small, acceptable cost).
- `|z|max` on watch: 39718 → ~3–5 (sane).
- Live prediction: constant → **varying** (it now responds to the signal).

### 3.2 Cross-device polarity / bias (second problem, partially handled)
Even after scale-fix, the subject's resting watch valence sat **strongly negative**
(measured resting median ≈ -0.55 to -0.65) — i.e., the model thinks the subject is
at rest "stressed". We treated this as a **measured neutral, not 0**, and built a
**personal valence baseline** (resting median+MAD, like the arousal baseline) so
`valence_personal = (valence_z − personal_median) / personal_MAD`. After
calibration, resting valence reads ~0 (neutral) as intended.

### 3.3 The live signal is dominated by noise (the wall we hit)
With the subject **sitting still, doing nothing emotional** (their own report):
- **Model confidence** `|2p−1|` averages **~0.57** (0.5 = pure chance). The model
  is barely above guessing on this subject.
- **Smoothed (60 s) personal valence** still has **std ≈ 0.5** on a [-1,1] scale
  while at rest — i.e., the "background mood" wanders ~half the full scale with no
  affective cause.
- The 2D emotion verdict (quadrant) flipped **~40% of epochs while sitting still**
  before stabilisation, i.e., it produced random-looking states.

We added the same machinery arousal uses to tame it:
- **Confidence gate** (epochs below ~0.7 confidence → treated as neutral),
- **median smoothing** over recent epochs,
- **verdict hysteresis** (quadrant changes only after N consecutive epochs),
- **subject-derived dead-band** (neutral zone width = a fraction of the subject's
  own resting spread, not a fixed guess).

This reduced flips to ~3%, but at a cost: **the subject's valence is "Neutral"
~50–80% of the time**, because the model is genuinely uncertain (~0.5 confidence).
That is arguably *honest* (report "don't know" rather than a fake quadrant), but it
means the valence axis carries almost no usable live information **for this
subject**.

### 3.4 Self-assessment of the subject (relevant confound)
The single test subject self-reports as a **low-responder** ("I don't laugh
easily, I don't get angry easily"). Peripheral valence signal scales with
affective reactivity/expressiveness; a low-responder produces small absolute
excursions, which is exactly where PPG valence is weakest (Section 2.4). We could
not run a proper validation session (induce known pleasant vs unpleasant states
and check the sign) because standard inductions (comedy/stressor clips) did not
produce reliable contrast in this subject.

### 3.5 Quantifying the device gap on the actual hardware (GalaxyPPG)
The single-subject failure in §3.1–3.3 was anecdotal. We quantified it on the
**GalaxyPPG** dataset (23 subjects, NeurIPS 2024) which records the *same Galaxy
Watch PPG as the product* simultaneously with Empatica E4 BVP and a Polar H10
chest ECG. (GalaxyPPG is stress-only, so it cannot teach valence — it is purely
a hardware diagnostic.) Two measurements, on rest segments:

- **Domain shift is total and normalisation does NOT close it.** A 5-fold
  classifier separating Galaxy-PPG windows from E4-BVP windows in the 33-feature
  space scores **AUC = 1.000** (perfect), with mean |Cohen's d| = 1.29 across
  features. Applying the §3.1 fix (`normalize_ppg_window`, median+MAD) leaves
  **AUC = 1.000** (mean |d| actually rises to 1.43). **This is the key finding:**
  scale-invariant normalisation removes the DC-offset/gain mismatch but the two
  sensors differ *fundamentally in pulse morphology* (Galaxy ~25 Hz DC-coupled
  optics vs E4 64 Hz AC-coupled) — a gap no amplitude normalisation can bridge.
  A model trained on E4 therefore sees watch PPG as an out-of-distribution signal;
  this is the rigorous explanation for the live constant-output / low-confidence
  collapse, beyond the single subject.
- **But the watch CAN sustain heart-rate / IBI.** Galaxy-PPG-derived IBI at rest
  has a median error of **17.5 ms (≈2.0 bpm)** vs Empatica's own R-R, essentially
  matching E4-BVP itself (**15.6 ms, ≈1.5 bpm**). The optical timing is good
  enough for HRV/arousal — which is exactly the modality we kept live. The device
  gap is specifically in the *morphological/vascular* features valence needs, not
  in the *rhythmic* features arousal needs.

**Net:** the hardware is adequate for arousal (rhythm) and inadequate for a
transferred valence model (morphology), and the cross-device morphology gap is
not fixable by normalisation alone — only by collecting watch-native training
data or per-subject adaptation.

---

## 4. Current decision

We **removed valence from the live watch verdict** and kept the watch on
**arousal only** (which works). The valence model, the personal calibration, the
verdict stabiliser, and all features remain in the codebase and still publish to a
research-only MQTT/Grafana channel for offline study — but they do **not** drive
any user-facing output. Valence is documented as an **explored direction with a
graded limitation**, not a delivered feature.

---

## 5. Questions for the reviewer (please be critical)

1. **Is the WESAD 82–85% LOSO meaningful, or is even this optimistic?** The 20 s
   windows with 10 s step overlap within a subject — does that leak within the
   training fold even under LOSO? Should windows be non-overlapping?
2. **Is "stress vs amusement" a valid valence proxy,** or are we still partly
   measuring arousal/effort differences rather than valence per se?
3. **The cross-device transfer:** is median+MAD per-window normalisation enough,
   or is the AC-coupled (E4) vs DC-coupled (watch) difference (plus different
   optical path, wavelength, contact pressure) a fundamental train/serve mismatch
   that no normalisation fixes? Would a small **per-subject fine-tune / few-shot
   adaptation** on watch data be the only honest path?
4. **The personal-baseline recentering:** does subtracting the subject's resting
   median *destroy* the valence signal (because resting affect IS the reference we
   want), or is it the correct way to remove cross-device bias? Where is the line?
5. **Confidence ~0.57 at rest:** is a model this uncertain ever usable with
   smoothing/gating, or is sub-0.6 margin proof the signal isn't there for this
   subject and we should stop?
6. **Low-responder confound:** is single-subject validation of a peripheral
   valence signal scientifically defensible at all, or does it inevitably reduce
   to "within-subject, low-N, uncontrolled" — the exact inflation we criticise in
   the literature?
7. **Is the honest framing** ("graded limitation: PPG valence is detectable
   proportional to induction strength; not usable live on a low-responder via
   wrist optics") **a sufficient and defensible thesis result,** or does it need a
   controlled induction experiment (e.g., cold-pressor for negative valence, which
   has a known vasoconstriction mechanism) to be credible?

---

## 6. Reproducibility pointers (for a reviewer with the repo)
- Training: `train/models/train_valence_wesad.py` (WESAD stress-vs-amusement, LOSO
  report, saves `models/valence_wesad.joblib`).
- Shared feature extractor (train/serve parity): `affectus/legacy/valence_features.py`
  (includes `normalize_ppg_window`, the scale-invariance fix).
- Feature families: `affectus/legacy/ppg_vascular_features.py`,
  `valence_ppg_fd.py`, `valence_ppg_morph.py`, `ppg_hrv_features.py`.
- Live channel + personal calibration: `affectus/shared/valence_baseline.py`,
  `affectus/shared/emotion.py` (verdict + stabiliser),
  `affectus/legacy/valence_wesad_engine.py`.
- Cross-dataset scripts: `train/extract/case_extract.py`,
  `train/extract/emowear_extract.py`; per-dataset results in `train/results/`.
