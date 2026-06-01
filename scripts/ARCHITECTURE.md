# scripts/

## Purpose
Offline / one-shot tooling: generate the Grafana dashboards and produce the
method-comparison report for the thesis. Not part of the live pipeline.

## Inputs
- `generate_grafana_dashboards.py`: nothing external - panel/SQL definitions live in code.
- `wesad_comparison_report.py`: reads InfluxDB (`biofizic_state` + `biofizic_legacy_wesad`).
- `validate_galaxyppg.py`: reads the replay trajectory from `tools/replay_galaxyppg.py`.

## Outputs
- **Dashboard JSON** written to `docker/grafana/provisioning/dashboards/*.json`
  (titles embed "what it shows - what to look for").
- **Comparison report**: Cohen's kappa (population vs personal labels), WESAD
  false-positive-rate-at-rest, Spearman rho - the numbers that justify the personal-baseline
  approach over a foreign-dataset ML model.
- **GalaxyPPG validation**: per-subject coverage + arousal behaviour vs Polar ECG
  ground truth, written to `eval_results/*.md`.

## Key files
| File | Role |
|---|---|
| `generate_grafana_dashboards.py` | `ts_panel`/`stat_panel`/`timeline_panel` builders + one `build_*_dashboard()` per board; `main()` writes the JSON files |
| `wesad_comparison_report.py` | Queries InfluxDB, computes kappa / FP-rate / rho |
| `validate_galaxyppg.py` | Validates the wrist-only pipeline against GalaxyPPG (21 subjects) vs Polar ECG |

## Data flow
```
panel defs (code) ─▶ generate_grafana_dashboards.py ─▶ provisioning/dashboards/*.json ─▶ Grafana
InfluxDB ─▶ wesad_comparison_report.py ─▶ κ + FP-rate + ρ (stdout / thesis)
```

## How to run
```
python scripts/generate_grafana_dashboards.py    # then: docker compose restart grafana
python scripts/wesad_comparison_report.py
```

## Depends on / Used by
- **Depends on:** InfluxDB (report only), `arousal_mapper.cohen_kappa`.
- **Used by:** Grafana (provisioned JSON), the thesis (the comparison numbers).
