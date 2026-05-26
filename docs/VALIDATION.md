# Biofizic validation runbook

What to run, in what order, to convince yourself (and the thesis committee)
that the pipeline does what the code claims.

There are four independent layers of validation. Run them in order; each
layer answers a different question.

## Layer 1 — Math unit tests (server)

**Question answered:** does the HRV pipeline produce the right number when
the inputs are known?

```bash
cd <repo-root>
python -m pytest tests/ -v
```

Expected: 29 tests pass in under 2 seconds. Covers IBI filtering, RMSSD /
SDNN / pNN50, Baevsky to Kubios SI conversion, zone boundaries (7.1 / 12.2
/ 22.4 / 30), the decision gate (alert confirmation, rest dual-veto, WALK
cap), the personal baseline (lock-in, persist, reset), and the live arousal
hysteresis (alternation immunity, sustained flip after N ticks).

## Layer 2 — Atomic sync unit tests (watch)

**Question answered:** are IBI and motion correctly anchored to the
same time window inside every acquisition/batch v2 payload?

```bash
cd <android-repo-root>
./gradlew :app:testDebugUnitTest
```

Expected: the `AcquisitionAssemblerTest` suite passes. Covers timestamp
normalization (ms / ns / invalid), IBI per-beat reconstruction walking
backwards from the anchor, `ts_anchor = max(ts_publish, IBI, skin temp)`,
the drain semantics (pending queue first, then 4.5 s horizon fallback), the
1 s motion stats window, the cardiac-band (0.5-4 Hz) motion energy, and the
full reset of `clear()`.

These tests are the proof that the atomic-sync code does what the docstring
of `AcquisitionAssembler` says it does.

## Layer 3 — Live diagnostic dashboard

**Question answered:** in production, with a real watch, are the streams
actually arriving together?

1. Bring up the stack:
   ```bash
   docker compose up -d
   ```
2. Start a tracking session on the watch.
3. Open Grafana at `http://localhost:3000`, dashboard
   **Biofizic Stream Sync Diagnostics** (UID `biofizic-stream-sync`).

What to look for:

| Panel | Healthy pattern |
|---|---|
| Atomic anchor delay (ts_anchor - ts_publish) | Mostly 0 to ~200 ms, occasionally up to ~500 ms when a stream's last sample lands slightly after the publish moment. Negative values would indicate a bug. |
| Sequence number | Strictly monotonic increment by 1, 1 per second. Plateaus or gaps indicate the publish loop is stalling. |
| IBI count per batch | Mostly 0, with spikes of 3 to 6 every ~4 s. Samsung Health groups HR/IBI in bursts; this pattern is *why* atomic sync exists. |
| Cardiac-band motion energy per batch | Near 0 at rest, rising with wrist motion in the 0.5-4 Hz band. This is the input the signal-quality gate uses to anticipate PPG artifacts. |
| Skin temp age at publish | A few seconds at most. Larger gaps mean the skin sensor stopped reporting. |

If any panel shows obviously wrong behaviour (negative anchor delays,
sequence going backwards), Layer 2 tests should catch it; if not, there is a
bug that needs new test coverage.

## Layer 4 — Offline reproducibility

**Question answered:** is the pipeline deterministic? Can you re-derive the
same decisions from the same input?

Record a session, then replay it into a clean stack and compare.

```bash
# 1. Record 5 minutes of a real session (watch must be publishing).
python tools/record_session.py \
    --broker localhost \
    --output sessions/baseline-rest.jsonl \
    --duration 300

# 2. Stop the watch, clear InfluxDB measurement for the replay window, then:
python tools/replay_session.py sessions/baseline-rest.jsonl --broker localhost

# 3. Open Grafana, biofizic-session-overview, and check that the replayed
#    arousal / Kubios SI / baseline z-score trace is the same shape as the
#    original recording.
```

The replayer rewrites `ts_publish`, `ts_anchor`, `ibi.ts` and `ppg.ts_ms` to
start at the current wall clock so Influx rows do not collide. Pass
`--keep-timestamps` to publish them unchanged (useful for byte-exact byte
comparison of a recording against a previous run).

## Layer 5 (optional) — External ground truth

For an evidence-grade comparison, export raw IBI from a recorded session into
[Kubios HRV](https://www.kubios.com/) (the de-facto reference desktop
software) and compare RMSSD / Kubios SI on matching 30 s windows. This is the
strongest claim you can make at thesis defence: not just "the code matches
its own spec" but "the code matches an external implementation of the same
spec". One paired-session comparison is enough; you do not need a large
dataset for a bachelor thesis.

To extract IBI from a session JSONL:

```bash
python -c "
import json, sys
for line in open(sys.argv[1]):
    rec = json.loads(line)
    ibi = rec['payload'].get('ibi') or {}
    for ms, ts in zip(ibi.get('ms', []), ibi.get('ts', [])):
        print(f'{ts},{ms}')
" sessions/baseline-rest.jsonl > sessions/baseline-rest.csv
```

The CSV columns are timestamp_ms,ibi_ms; import into Kubios as "Beat-to-beat
RR interval".

---

## Integrity / correctness framework

How we argue the chain (IBI → HRV → personal z → arousal) is correct, not just
plausible:

1. **Golden synthetic signals** (`tests/test_golden_signals.py`): an IBI series
   alternating M±d has analytic RMSSD=2d, SDNN=d, HR=60000/M. The pipeline math
   is asserted against these exact targets, and the artifact corrector is shown
   to recover the clean RMSSD after a spike is injected.
2. **Reference oracle** (`tests/test_reference_hrv.py`, dev-only): RMSSD/SDNN are
   cross-checked against NeuroKit2 on the same IBI (skipped if NeuroKit2 is not
   installed; it is not a production dependency).
3. **Determinism**: replaying a recorded session reproduces identical decisions.
4. **Quality introspection**: `biofizic/state` exposes `artifact_rate`,
   `signal_quality`, `z_si` (raw), `z_si_filtered` and `kalman_gain`, so a wrong
   output can be attributed to a noisy input (low quality, low gain) rather than
   a wrong algorithm.

### Controlled physiological protocol (V.4)

Run a scripted session and check the production arousal follows the expected
qualitative trajectory (it is the empirical validity check):

| Phase | Duration | Expected physiology | Expected output |
|-------|----------|--------------------|-----------------|
| Rest (seated, quiet) | 5 min | baseline RMSSD/SI; baseline locks | arousal settles mid/low, `kalman_gain` healthy |
| Paced breathing 6/min | 2 min | RMSSD ↑ (vagal), SI ↓ | arousal trends **down** (z_filtered negative) |
| Cognitive stressor (e.g. mental arithmetic / Stroop) | 2 min | RMSSD ↓, SI ↑ | arousal trends **up**; CUSUM may fire `alert` |
| Recovery (quiet) | 3 min | return toward baseline | arousal returns toward baseline |

Record it (`tools/record_session.py`) so the same run can be replayed and shown
in Grafana for the defence.

## Quick checklist before thesis defence

- [ ] `python -m pytest tests/ -v` runs in under 2 seconds, all green
- [ ] `./gradlew :app:testDebugUnitTest` runs all green
- [ ] During a live session, `biofizic-stream-sync` shows positive anchor
      delays and monotonic seq
- [ ] At least one recorded session can be replayed and reproduces the same
      decisions
- [ ] The controlled protocol (rest → paced breathing → stressor → recovery)
      produces the expected arousal trajectory
- [ ] At least one paired comparison against Kubios HRV (optional but nice)
