# biofizic/ (package root)

## Purpose
The Python domain package: every compute module lives here. The root itself holds the
cross-cutting pieces — central configuration, the human-readable log format, and the
import bootstrap — that all sub-packages depend on.

## Inputs
- Nothing at runtime; these are imported by everything else.
- `config.py` is the single source of truth for constants (thresholds, window lengths,
  Kalman/CUSUM params, Kubios zone bounds), each with a provenance comment (literature
  or infrastructure).

## Outputs
- Constants and helpers consumed by `ingestion`, `dsp`, `compute_features`, `engine`,
  `legacy`, and `services`.
- `logging.py` turns a `PhysiologyDecision` into the `[SUMMARY]/[BASELINE]/[QUALITY]/[WINDOW]`
  block seen in `docker compose logs compute-engine`.

## Key files
| File | Role |
|---|---|
| `config.py` | All tunable constants + their scientific/infra justification |
| `logging.py` | Decision → multi-line human log (one factor per line) |
| `_bootstrap.py` | Import-path / package bootstrap |
| `__init__.py` | Package marker |

## Data flow
```
config.py ──(constants)──▶ every sub-package
PhysiologyDecision ──▶ logging.py ──▶ stdout (Docker logs)
```

## Depends on / Used by
- **Depends on:** nothing internal (leaf).
- **Used by:** all sub-packages + `services/`.
- Note: a subset of `config.py` constants (IBI physiological band) is duplicated by hand
  on the watch (`signal/IbiPipeline.kt`) and must stay in sync.
