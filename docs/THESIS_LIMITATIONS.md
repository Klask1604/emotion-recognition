# Limitari si parametri empirici (Biofizic, teza)

What is validated by literature versus chosen empirically, and a record of
the alternatives that were tried and rejected. Both lists are deliberately
short and concrete so they can be cited directly in the thesis chapter on
limitations.

## Validated by literature

- IBI filter 300 to 2000 ms with median outlier rejection at 20 percent
- RMSSD as a parasympathetic marker
- Baevsky Stress Index, then Kubios SI = sqrt(SI), with zones 7.1 / 12.2 / 22.4 / 30.0 from the Kubios HRV User Guide
- Personal baseline as a robust log-space z-score: RMSSD and SI are log-normal
  (Task Force ESC/NASPE 1996; Nunan 2010), estimated with median + MAD
  (sigma = 1.4826*MAD, Hampel). Arousal = Phi(z) via the normal CDF.
- Signal-quality gate instead of activity classification: IBI artifact rate is a
  standard HRV quality index (Task Force 1996), and cardiac-band (0.5-4 Hz)
  acceleration energy is the motion that overlaps and corrupts the PPG pulse
  (accelerometer motion-artifact removal, TROIKA, Zhang 2015).
- CUSUM sequential change detection for sustained-stress alerts (Page 1954;
  Montgomery SPC), with z already in sigma units so k and h take SPC defaults.
- Fixed decision window w30; w60 and w90 are kept only for validation (see below)

## Empirical parameters (tuned on Galaxy Watch 7, not from literature)

| Parameter | Value | Role |
|-----------|-------|------|
| `BASELINE_MIN_REST_EPOCHS` | 12 | Resting epochs (~6 min) before the baseline locks; fewer is too noisy for a stable MAD |
| `BASELINE_ROBUST_WINDOW_EPOCHS` | 60 | Rolling window the median/MAD slides over (adapts to circadian drift without an EMA) |
| `BASELINE_LOG_SIGMA_FLOOR` | 0.05 | Numerical floor on the log-space sigma (INFRA) |
| `ARTIFACT_RATE_MAX` | 0.05 | Reliability cutoff on IBI artifact rate, aligned with the Kubios "low" correction regime |
| `QUALITY_LOGISTIC_LEARNING_RATE` | 0.05 | SGD step for the per-user P(artifact \| motion) logistic fit (INFRA) |
| `CUSUM_SLACK_K` / `CUSUM_THRESHOLD_H` | 0.5 / 4.0 | SPC defaults; z is already in sigma units |
| `LIVE_AROUSAL_HYSTERESIS_TICKS` | 3 | Consecutive live ticks required before the watch UI adopts a new integer. Two ticks of agreement turned out to still let pure 2-tick alternations through. |

All other constants live in `biofizic/config.py` with a one-line provenance
comment (LITERATURE / EMPIRICAL / INFRA) next to each section.

## Hardware limitations

- Kubios zones are calibrated on clinical ECG and Polar H10 chest straps. The Galaxy Watch 7 uses an optical wrist PPG, so there is a measurable domain shift.
- The Samsung Health SDK batches PPG samples when the watch screen is off. The cadence is irregular (8 to 15 s gaps) and the timestamps inside a batch are only approximately uniform, which collapses any narrow-band frequency analysis. The thesis pipeline does not consume raw PPG anymore for that reason.

## Motion: signal quality, not activity classification

The pipeline no longer classifies activities at all. The WISDM Random Forest
HAR classifier was retired for a concrete, demonstrable reason: a train/serve
feature mismatch. It was trained on 20 Hz WISDM windows whose last two features
were FFT band energies, but at inference on the Galaxy Watch 7 it received the
watch's 1 Hz aggregated motion statistics, with those same two columns filled by
crude linear rescalings of acceleration — and a per-user normaliser was applied
at inference although the model had been fit on raw features. The two feature
spaces did not match, so the classifier's output on-device was not trustworthy.

The only role HAR ever played was to cap arousal when wrist motion corrupts the
PPG. That physical cause is now measured directly by the signal-quality gate
(`decision/signal_quality.py`): the IBI artifact rate plus the cardiac-band
(0.5-4 Hz) acceleration energy reported by the watch, with the motion->artifact
relationship learned per subject by online logistic regression. A coarse
still/moving label is derived from the learned threshold for the UI only; it is
not an input to any decision.

(Earlier versions also ran a `context_engine` and an `AdaptiveMotionBaseline`
1-component GMM in parallel; both were removed previously as redundant — a
single-component GMM is just a mean and standard deviation, so depending on
`scikit-learn` for it was overkill. `scikit-learn` and `joblib` are no longer
dependencies.)

## What was tried and rejected (thesis "Limitations" chapter)

### Raw PPG processing on the server

A standalone `ppg-processor` service ran in parallel with the IBI pipeline:
Butterworth band-pass (0.5 to 4 Hz), three peak detectors (`scipy.find_peaks`
on positive and inverted signal, plus NeuroKit2 Elgendi) with the most
plausible candidate selected by an IBI-validity score, and a personal
amplitude baseline. It published a single feature, `z_pulse_amp`, intended as
a sympathetic proxy.

Findings that motivated the rejection:

1. Screen-off batching by the Samsung SDK left 8 to 15 second gaps in PPG
   data. The 30 s analysis window could not be filled with continuous samples
   often enough to make the output trustworthy.
2. Wrist motion above ~2 m/s^2 contaminates the optical signal in a way no
   filter recovers; the motion gate had to discard most epochs.
3. The only consumer of `z_pulse_amp` was the experimental valence proxy
   (also removed, see below), which means the entire 750 lines of DSP fed
   nothing the watch UI actually showed.

For the thesis: the conclusion is that wrist PPG on a consumer smartwatch is
usable for HR and IBI (with the SDK doing peak detection on-device) but is
not a viable source for autonomic features beyond what HRV from IBI already
provides.

### Multi-window HRV (w30 vs w60 vs w90)

Three analysis windows are computed on every tick and published on
`biofizic/state/windows` plus the `biofizic-window-comparison` dashboard. They
are not used for decisions; the only decision window is w30.

The reason for keeping the three side by side is to justify the choice of
w30 against w60 and w90 in the thesis:

- w30 reacts fast enough to catch a sustained sympathetic shift inside a
  single epoch boundary, which is what the watch UI is supposed to show.
- w60 and w90 are smoother (better confidence per metric) but they smear
  transient stress responses over their longer window, and on a watch that
  already publishes only every 30 s they delay feedback to the user by an
  unacceptable amount.
- Side-by-side traces during a session show RMSSD and Kubios SI from all
  three windows converging when the subject is at rest, and diverging during
  the first ~30 s of a stress response, which is the empirical justification
  for using w30 as the primary window.

### Valence and affect quadrant

`valence_mapper.py` combined RMSSD z-score and PPG amplitude z-score with
coefficients 0.55 and 0.35 to produce a 1 to 10 valence axis, then mapped
arousal x valence onto Russell's circumplex (calm / activated / tense /
depleted).

Removed for two reasons:

1. The coefficients were tuned ad-hoc on a handful of recordings and have no
   support in the HRV literature. Calling that "valence" would have been
   dishonest at thesis defence.
2. Wrist PPG amplitude correlates with vasoconstriction and skin
   temperature; it does not separate positive from negative affect.

The Russell circumplex itself is a valid psychological model. The data
collected with this hardware is not enough to support the valence axis, so
the affect quadrant view was removed from production. The thesis keeps it as
a documented attempt, not as a result.
