# scripts/

## Purpose
Offline / one-shot tooling: generate the Grafana dashboards and produce the
method-comparison report for the thesis. Not part of the live pipeline.

## Inputs
- `generate_grafana_dashboards.py`: nothing external — panel/SQL definitions live in code.
- `wesad_comparison_report.py`: reads InfluxDB (`biofizic_state` + `biofizic_legacy_wesad`).

## Outputs
- **Dashboard JSON** written to `docker/grafana/provisioning/dashboards/*.json`
  (13 dashboards; titles embed "what it shows — what to look for").
- **Comparison report**: Cohen's κ (population vs personal labels), WESAD
  false-positive-rate-at-rest, Spearman ρ — the numbers that justify the personal-baseline
  approach over a foreign-dataset ML model.

## Key files
| File | Role |
|---|---|
| `generate_grafana_dashboards.py` | `ts_panel`/`stat_panel`/`timeline_panel` builders + one `build_*_dashboard()` per board; `main()` writes the JSON files |
| `wesad_comparison_report.py` | Queries InfluxDB, computes κ / FP-rate / ρ |
| `export_session_influx.py` | Export a session (state + combined + sensors) from InfluxDB to CSV |
| `generate_ml_evaluation_report.py` | Build a deterministic ML evaluation report from `eval_results/*.json` |

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
