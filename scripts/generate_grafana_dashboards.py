#!/usr/bin/env python3
"""Generate Grafana dashboard JSON files for Biofizic."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docker" / "grafana" / "provisioning" / "dashboards"

DS = {"type": "influxdb", "uid": "biofizic-influx"}

# Map Romanian display labels to numeric levels for timeseries comparison.
LABEL_LEVEL_SQL = """
CASE {col}
  WHEN 'Relaxat' THEN 1
  WHEN 'Echilibrat' THEN 2
  WHEN 'Moderat' THEN 3
  WHEN 'Alert' THEN 4
  WHEN 'Ridicat' THEN 5
  ELSE 0
END
""".strip()


def dashboard_meta(uid: str, title: str, tags: list[str], panels: list, *, version: int = 1, refresh: str = "5s") -> dict:
    """Standard dashboard envelope; id=null lets Grafana assign stable identity by uid."""
    return {
        "id": None,
        "uid": uid,
        "title": title,
        "tags": tags,
        "timezone": "browser",
        "schemaVersion": 39,
        "version": version,
        "refresh": refresh,
        "panels": panels,
    }


def ds_target(sql: str, ref: str = "A", fmt: str = "time_series") -> dict:
    return {
        "datasource": DS,
        "editorMode": "code",
        "format": fmt,
        "queryType": "sql",
        "rawQuery": True,
        "rawSql": sql,
        "refId": ref,
    }


def _ts_defaults(*, unit: str | None = None, decimals: int = 1) -> dict:
    defaults: dict = {
        "color": {"mode": "palette-classic"},
        "custom": {
            "drawStyle": "line",
            "lineWidth": 2,
            "fillOpacity": 10,
            "showPoints": "never",
            "spanNulls": True,
            "lineInterpolation": "smooth",
        },
        "decimals": decimals,
    }
    if unit:
        defaults["unit"] = unit
    return defaults


def ts_panel(
    panel_id: int,
    title: str,
    y: int,
    sql: str,
    *,
    h: int = 8,
    w: int = 24,
    x: int = 0,
    extra_sql: list[str] | None = None,
    unit: str | None = None,
    min_v: float | None = None,
    max_v: float | None = None,
    decimals: int = 1,
) -> dict:
    targets = [ds_target(sql)]
    for i, extra in enumerate(extra_sql or [], start=1):
        targets.append(ds_target(extra, ref=chr(ord("A") + i)))
    defaults = _ts_defaults(unit=unit, decimals=decimals)
    if min_v is not None:
        defaults["min"] = min_v
    if max_v is not None:
        defaults["max"] = max_v
    return {
        "id": panel_id,
        "title": title,
        "type": "timeseries",
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": targets,
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }


def stat_panel(
    panel_id: int,
    title: str,
    sql: str,
    x: int,
    y: int,
    *,
    w: int = 6,
    unit: str | None = None,
    decimals: int = 1,
) -> dict:
    defaults: dict = {
        "decimals": decimals,
        "color": {"mode": "thresholds"},
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
    }
    if unit:
        defaults["unit"] = unit
    return {
        "id": panel_id,
        "title": title,
        "type": "stat",
        "gridPos": {"h": 4, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds_target(sql, fmt="table")],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "/^v$/", "values": False},
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "none",
        },
    }


def timeline_panel(
    panel_id: int,
    title: str,
    sql: str,
    y: int,
    *,
    h: int = 6,
    w: int = 24,
    x: int = 0,
) -> dict:
    return {
        "id": panel_id,
        "title": title,
        "type": "state-timeline",
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds_target(sql, fmt="table")],
        "fieldConfig": {
            "defaults": {"custom": {"fillOpacity": 75, "lineWidth": 0}},
            "overrides": [],
        },
        "options": {
            "mergeValues": True,
            "rowHeight": 0.85,
            "showValue": "auto",
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        },
    }


def label_level_sql(column: str, alias: str) -> str:
    return LABEL_LEVEL_SQL.format(col=column).replace("\n", " ") + f" AS {alias}"


def build_hrv_dashboard() -> dict:
    panels = [
        stat_panel(
            1,
            "RMSSD 30s",
            "SELECT rmssd AS v FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time DESC LIMIT 1",
            0,
            0,
            unit="ms",
        ),
        stat_panel(
            2,
            "Stress index",
            "SELECT stress_index AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            decimals=2,
        ),
        stat_panel(
            3,
            "Label Kubios",
            "SELECT emotion AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            decimals=0,
        ),
        stat_panel(
            4,
            "Label baseline",
            "SELECT emotion_baseline AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18,
            0,
            decimals=0,
        ),
        ts_panel(
            5,
            "RMSSD multi-window (15/30/60/90 s)",
            4,
            "SELECT time, rmssd_w30, rmssd_w60, rmssd_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd_w15 FROM biofizic_state WHERE $__timeFilter(time) AND rmssd_w15 > 0 ORDER BY time",
            ],
            unit="ms",
        ),
        ts_panel(
            6,
            "Stress index multi-window",
            12,
            "SELECT time, stress_index_w30, stress_index_w60, stress_index_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, stress_index_w15 FROM biofizic_state WHERE $__timeFilter(time) AND stress_index_w15 > 0 ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            7,
            "Heart rate vs RMSSD",
            20,
            "SELECT time, mean_hr AS hr_bpm FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd AS rmssd_ms FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            unit="none",
        ),
    ]
    return dashboard_meta("biofizic-hrv-analysis", "Biofizic HRV Analysis", ["biofizic", "hrv"], panels, version=3)


def build_baseline_dashboard() -> dict:
    label_levels = (
        "SELECT time, "
        f"{label_level_sql('emotion', 'kubios_label')}, "
        f"{label_level_sql('emotion_baseline', 'baseline_label')} "
        "FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time"
    )
    panels = [
        timeline_panel(
            1,
            "Kubios vs baseline labels (text)",
            "SELECT time, emotion, emotion_baseline FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            0,
        ),
        ts_panel(
            2,
            "Label levels (1=Relaxat … 5=Ridicat) + agreement",
            6,
            label_levels,
            extra_sql=[
                "SELECT time, labels_agree * 5 AS labels_agree_scaled FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
            h=7,
            min_v=0,
            max_v=5,
            decimals=0,
        ),
        ts_panel(
            3,
            "Stress index vs personal baseline",
            13,
            "SELECT time, stress_index AS si_kubios FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, baseline_si AS si_baseline FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            4,
            "Stress index z-score",
            21,
            "SELECT time, z_si FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=-3,
            max_v=3,
            decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-baseline-compare", "Biofizic Baseline Compare", ["biofizic", "baseline"], panels, version=3
    )


def build_motion_dashboard() -> dict:
    panels = [
        stat_panel(
            1,
            "HAR class",
            "SELECT motion_class AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0,
            0,
            w=6,
            decimals=0,
        ),
        stat_panel(
            2,
            "Activity mode",
            "SELECT activity_mode AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            w=6,
            decimals=0,
        ),
        stat_panel(
            3,
            "Motion confidence",
            "SELECT motion_conf AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            w=6,
            decimals=2,
        ),
        timeline_panel(
            4,
            "HAR class over time",
            "SELECT time, motion_class FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            4,
            h=5,
        ),
        ts_panel(
            5,
            "Motion confidence",
            9,
            "SELECT time, motion_conf FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            min_v=0,
            max_v=1,
            decimals=2,
        ),
        ts_panel(
            6,
            "Acceleration stats (1 Hz batch)",
            16,
            "SELECT time, acc_rms, acc_p90, gyro_rms FROM biofizic_acquisition_batch WHERE $__timeFilter(time) ORDER BY time",
            unit="accMS2",
        ),
    ]
    return dashboard_meta("biofizic-motion-har", "Biofizic Motion HAR", ["biofizic", "motion"], panels, version=4)


def build_overview_dashboard() -> dict:
    """Live overview from biofizic_state_live (1 Hz). Epoch decisions remain on biofizic_state."""
    panels = [
        stat_panel(
            1,
            "Arousal (live)",
            "SELECT arousal_10 AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time DESC LIMIT 1",
            0,
            0,
            decimals=0,
        ),
        stat_panel(
            2,
            "Kubios label",
            "SELECT emotion AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            decimals=0,
        ),
        stat_panel(
            3,
            "IBI buffer size",
            "SELECT ibi_buffer_size AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            decimals=0,
        ),
        stat_panel(
            4,
            "Data quality",
            "SELECT data_quality AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18,
            0,
            decimals=0,
        ),
        ts_panel(
            5,
            "Arousal (live 1 Hz)",
            4,
            "SELECT time, arousal_10 FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time",
            min_v=0,
            max_v=10,
            decimals=0,
        ),
        ts_panel(
            6,
            "RMSSD and stress index (live)",
            12,
            "SELECT time, rmssd, stress_index FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            decimals=2,
        ),
        timeline_panel(
            7,
            "Data quality over time",
            "SELECT time, data_quality FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time",
            20,
            h=4,
        ),
        ts_panel(
            8,
            "Skin temp and HR (1 Hz batch)",
            24,
            "SELECT time, skin_temp FROM biofizic_acquisition_batch WHERE $__timeFilter(time) AND skin_temp > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, hr FROM biofizic_acquisition_batch WHERE $__timeFilter(time) AND hr > 0 ORDER BY time",
            ],
            h=6,
        ),
    ]
    return dashboard_meta(
        "biofizic-live-overview",
        "Biofizic Live Overview",
        ["biofizic", "overview"],
        panels,
        version=6,
        refresh="1s",
    )


def build_window_comparison_dashboard() -> dict:
    """Compare HRV metrics across 30/60/90s windows from biofizic_state_windows."""
    panels = [
        ts_panel(
            1,
            "RMSSD per window",
            0,
            "SELECT time, w30_rmssd, w60_rmssd, w90_rmssd FROM biofizic_state_windows "
            "WHERE $__timeFilter(time) AND w30_rmssd > 0 ORDER BY time",
            unit="ms",
        ),
        ts_panel(
            2,
            "Stress index per window",
            8,
            "SELECT time, w30_stress_index, w60_stress_index, w90_stress_index "
            "FROM biofizic_state_windows WHERE $__timeFilter(time) ORDER BY time",
            decimals=2,
        ),
        ts_panel(
            3,
            "IBI count per window",
            16,
            "SELECT time, w30_ibi_count, w60_ibi_count, w90_ibi_count "
            "FROM biofizic_state_windows WHERE $__timeFilter(time) ORDER BY time",
            decimals=0,
        ),
        stat_panel(
            4,
            "IBI buffer size",
            "SELECT ibi_buffer_size AS v FROM biofizic_state_windows "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0,
            24,
            w=8,
            decimals=0,
        ),
        timeline_panel(
            5,
            "Window quality (w30 / w60 / w90)",
            "SELECT time, w30_quality, w60_quality, w90_quality "
            "FROM biofizic_state_windows WHERE $__timeFilter(time) ORDER BY time",
            28,
            h=5,
        ),
    ]
    return dashboard_meta(
        "biofizic-window-comparison",
        "Biofizic Window Comparison",
        ["biofizic", "windows"],
        panels,
        version=1,
        refresh="5s",
    )


def build_session_overview_dashboard() -> dict:
    panels = [
        ts_panel(
            1,
            "Heart rate and RMSSD",
            0,
            "SELECT time, mean_hr AS hr_bpm FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd AS rmssd_ms FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            h=7,
        ),
        ts_panel(
            2,
            "Stress index and z-score",
            7,
            "SELECT time, stress_index, z_si FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            decimals=2,
        ),
        timeline_panel(
            3,
            "Motion class",
            "SELECT time, motion_class FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            14,
            h=4,
        ),
        ts_panel(
            4,
            "Acceleration RMS (1 Hz batch)",
            18,
            "SELECT time, acc_rms FROM biofizic_acquisition_batch WHERE $__timeFilter(time) ORDER BY time",
            h=6,
            unit="accMS2",
        ),
        ts_panel(
            5,
            "Arousal",
            24,
            "SELECT time, arousal_10 FROM biofizic_state "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time",
            min_v=0,
            max_v=10,
            decimals=0,
        ),
        ts_panel(
            6,
            "Label agreement (Kubios vs baseline)",
            32,
            "SELECT time, labels_agree * 5 AS labels_agree_scaled FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=0,
            max_v=5,
            decimals=0,
            h=5,
        ),
    ]
    return dashboard_meta(
        "biofizic-session-overview", "Biofizic Session Overview", ["biofizic", "session"], panels, version=3
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dashboards = {
        "biofizic-hrv-analysis.json": build_hrv_dashboard(),
        "biofizic-baseline-compare.json": build_baseline_dashboard(),
        "biofizic-motion-har.json": build_motion_dashboard(),
        "biofizic-live-overview.json": build_overview_dashboard(),
        "biofizic-window-comparison.json": build_window_comparison_dashboard(),
        "biofizic-session-overview.json": build_session_overview_dashboard(),
    }
    for name, body in dashboards.items():
        path = OUT / name
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
