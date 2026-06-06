# Architecture

The single source of truth for how the system works. Written from the code, not
from memory. Check here before claiming anything about the system.

## The flow, end to end

```
Watch (wrist) --WiFi--> MQTT --> Python engine (Docker) --> InfluxDB --> Grafana
                                      |
                                      +--> verdict back to the watch
                                      +<--> Unity / VR (scene context + verdict)
```

Chest and headset devices are not implemented. The capability contract for them
exists in `affectus/contract/` (DDD), so when they are added they declare what
signals they carry through the handshake and get a channel in `engine/channels.py`.
For now only the wrist is real and tested.

## What the watch sends

One batch per second on `biofizic/acquisition/batch`. Parsed in
`compute_engine.py`. It carries:

- Cardiac: IBI (intervals + timestamps), HR in bpm. Raw 100 Hz PPG comes separately
  on `biofizic/ppg/ondemand`.
- Temperature: skin and ambient.
- Motion: accelerometer (rms, p90, std, cardiac band) and gyroscope (rms, p90, std).

Motion is not an emotion signal. It is a quality gate. It detects movement, drops
contaminated windows, and decides when the user is still, so the baseline is learned
only at rest.

## Timing

From `config.py`:

| Constant | Value | Meaning |
|---|---|---|
| Batch from watch | ~1 s | one message per second |
| Analysis windows | 30, 60, 90 s | HRV windows computed in parallel |
| Primary window | 60 s | the window that drives the decision |
| Epoch interval | 30 s | the official verdict fires every 30 s |
| Live | ~1 s | fast state on `biofizic/state/live` |

Window vs epoch, do not confuse them. The window is how much data it looks back on
(60 s). The epoch is how often it commits an official verdict (30 s). So every 30 s
it takes the last 60 s of data and produces a verdict.

## Axis 1: arousal

In `engine/decision.py`. This is the part that works and is validated.

1. From HRV (RMSSD, SDNN, Baevsky/Kubios stress index) plus skin temperature.
2. Personal z-score against the user's own resting baseline, not a population mean.
   This personal baseline is the single biggest lever: it is the difference between
   chance and a working verdict.
3. Fuse the HRV and HR channels by signal quality. HRV is precise when still, HR is
   robust when moving. The weight is the quality Q.
4. Scalar Kalman filter, once per epoch, smooths the estimate so it does not jump.
5. CUSUM change gate flags sudden shifts. It is published as an alert flag but does
   not change the verdict itself.
6. Map to a 0 to 10 scale through a personal CDF when calibrated, or a population
   zone before the baseline locks.

The arousal baseline locks after about 6 minutes of resting epochs. It survives
restarts (persisted to disk), so the engine does not recalibrate every time.

## Axis 2: valence

Valence does not come from the pulse rate the way arousal does. HRV cannot separate
pleasant from unpleasant (it sits at chance, around 52% on disconfort vs placut).

What does work is PPG pulse-shape morphology. The shape of the wave (amplitude, rise
time, width, area) carries vasoconstriction, which tracks valence. On WESAD at 64 Hz
this separates disconfort from placut at 86%, where HRV is at chance.

The morphology model is in `research/stari/morpho_classifier.py`. It consumes the
100 Hz PPG, downsamples to 64 Hz to match training exactly, extracts the shape
features, and applies a personal baseline. The features are in real units
(milliseconds, amplitude), so they transfer across sampling rates without distortion.

One caveat the code is honest about: morphology only separates valence well when the
user is activated. At low arousal it is not validated, so the verdict marks valence
as unreliable there (`valence_reliable=false`).

## The verdict: Russell quadrant

In `_publish_emotion_state` in `compute_engine.py`, combining the two axes in a fixed
order (arousal first):

1. Arousal decides calm vs activated. This is the reliable part, so it goes first.
   The state classifier trained on film stimuli cannot recognise calm reliably, so we
   do not ask it. Below the activation threshold the verdict is calm, straight from
   arousal.
2. Only when activated, the morphology model decides disconfort vs placut.

The output is a quadrant plus the typical emotions for it, not an exact emotion:

| Zone | Typical emotions |
|---|---|
| Activated, positive | Entuziasm, Bucurie, Excitare |
| Activated, negative | Stres, Furie, Anxietate |
| Calm / deactivated | Relaxare, Odihna, Plictiseala |
| Neutral (mid arousal) | Concentrat, Atent, Echilibrat |

The Russell coordinates are published too: `arousal_y` (0 to 10) and `valence_x`
(-1 to +1). When the morphology model votes calm, `valence_x` is forced to 0 so calm
does not produce a fake valence sign.

## The two state models

Both in `research/stari/`, both trained on the watch sensor, both with a personal
baseline learned live and persisted to disk.

- `state_classifier.py`: HRV features (rmssd, mean_hr, sdnn, mean_ibi, pnn50) plus
  motion. Strong at calm vs activated (97%).
- `morpho_classifier.py`: PPG pulse-shape features. Strong at the valence direction
  when activated (86%).

The feature vectors are chosen to match exactly what the live engine produces, so
there is no train-vs-serving gap.

## Research engines (not in the verdict)

`research/` holds the older valence work (FD, vascular, the WESAD/EEVR/CASE valence
models, respiration, polarity). They run in parallel and publish on `biofizic/legacy/*`
for the Grafana dashboards, but they never feed the production verdict. They are kept
on because their dashboards are evidence for the thesis (valence comes out near chance,
which is the point). Toggled in `research/toggles.py`. See `research/README.md` for what
each one does and `docs/research-log/` for the full history.

## Topics

The full MQTT topic list is in `docs/TOPICS.md`.

## Implementation status

- Wrist: implemented and tested. The only part validated on a real subject.
- Chest, headset: capability contract ready in `contract/`, devices not implemented.
- Unity / VR: scene context to valence is partial. The live feedback loop is not done.
