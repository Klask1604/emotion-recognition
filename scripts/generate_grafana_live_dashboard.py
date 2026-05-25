#!/usr/bin/env python3
"""Genereaza docker/grafana/provisioning/dashboards/biofizic-live.json"""

import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "biofizic-influx"}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docker/grafana/provisioning/dashboards/biofizic-live.json"


def ds_target(sql: str, ref: str = "A", fmt: str = "time_series") -> dict:
    return {
        "datasource": DS,
        "editorMode": "code",
        "format": fmt,
        "rawQuery": True,
        "rawSql": sql,
        "refId": ref,
    }


def row_panel(title: str, y: int, id_: int) -> dict:
    return {
        "id": id_,
        "type": "row",
        "title": title,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "collapsed": False,
        "panels": [],
    }


def ts_panel(
    pid: int,
    title: str,
    y: int,
    x: int,
    w: int,
    h: int,
    sql: str,
    extra_targets: list[str] | None = None,
    unit: str | None = None,
    min_v: float | None = None,
    max_v: float | None = None,
    decimals: int = 1,
) -> dict:
    targets = [ds_target(sql)]
    for i, s in enumerate(extra_targets or [], start=1):
        targets.append(ds_target(s, ref=chr(ord("A") + i)))
    p = {
        "id": pid,
        "type": "timeseries",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": targets,
        "fieldConfig": {
            "defaults": {
                "color": {"mode": "palette-classic"},
                "custom": {
                    "drawStyle": "line",
                    "lineWidth": 2,
                    "fillOpacity": 12,
                    "showPoints": "never",
                    "spanNulls": True,
                    "lineInterpolation": "smooth",
                },
                "decimals": decimals,
            },
            "overrides": [],
        },
        "options": {
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
            "tooltip": {"mode": "multi", "sort": "none"},
        },
    }
    d = p["fieldConfig"]["defaults"]
    if unit:
        d["unit"] = unit
    if min_v is not None:
        d["min"] = min_v
    if max_v is not None:
        d["max"] = max_v
    return p


def stat_panel(
    pid: int,
    title: str,
    y: int,
    x: int,
    w: int,
    sql: str,
    unit: str | None = None,
    decimals: int = 0,
) -> dict:
    defaults: dict = {
        "decimals": decimals,
        "color": {"mode": "thresholds"},
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
    }
    if unit:
        defaults["unit"] = unit
    if unit == "percent":
        defaults["max"] = 100
    return {
        "id": pid,
        "type": "stat",
        "title": title,
        "gridPos": {"h": 4, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds_target(sql, fmt="table")],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "orientation": "auto",
            "textMode": "auto",
            "colorMode": "value",
            "graphMode": "none",
        },
    }


def timeline_panel(pid: int, title: str, y: int, x: int, w: int, h: int, sql: str) -> dict:
    return {
        "id": pid,
        "type": "state-timeline",
        "title": title,
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


def main() -> None:
    panels: list[dict] = []
    y = 0

    panels.append(row_panel("Rezumat live", y, 100))
    y += 1

    stats = [
        (1, "HR live", "SELECT hr AS v FROM biofizic_hr_live WHERE $__timeFilter(time) AND hr > 0 ORDER BY time DESC LIMIT 1", "bpm"),
        (2, "RMSSD Samsung", "SELECT rmssd AS v FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time DESC LIMIT 1", "ms"),
        (3, "Arousal %", "SELECT arousal_pct AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", "percent"),
        (4, "Arousal /10", "SELECT arousal_10 AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
        (5, "Eticheta", "SELECT emotion AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
        (6, "Mod activitate", "SELECT activity_mode AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
    ]
    for i, (pid, title, sql, unit) in enumerate(stats):
        panels.append(stat_panel(pid, title, y, i * 4, 4, sql, unit))
    y += 4

    panels.append(row_panel("Cardiac & HRV — comparatii", y, 101))
    y += 1
    panels.append(
        ts_panel(
            10,
            "HR: live vs epoch vs state",
            y,
            0,
            12,
            8,
            "SELECT time, hr AS hr_live FROM biofizic_hr_live WHERE $__timeFilter(time) AND hr > 0 ORDER BY time",
            [
                "SELECT time, mean_hr AS hr_epoch FROM biofizic_epoch WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
                "SELECT time, mean_hr AS hr_state FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            ],
            unit="bpm",
            min_v=45,
            max_v=140,
        )
    )
    panels.append(
        ts_panel(
            11,
            "RMSSD: Samsung vs PPG vs epoch",
            y,
            12,
            12,
            8,
            "SELECT time, rmssd AS rmssd_samsung FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            [
                "SELECT time, rmssd_ppg FROM biofizic_ppg_hrv WHERE $__timeFilter(time) AND rmssd_ppg > 0 ORDER BY time",
                "SELECT time, rmssd AS rmssd_epoch FROM biofizic_epoch WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            unit="ms",
            min_v=0,
            max_v=250,
        )
    )
    y += 8

    panels.append(row_panel("Activare fiziologica (Baevsky + percentil)", y, 102))
    y += 1
    panels.append(
        ts_panel(
            20,
            "Stress index & Baevsky SI",
            y,
            0,
            12,
            8,
            "SELECT time, stress_index, baevsky_si FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            decimals=2,
        )
    )
    panels.append(
        ts_panel(
            21,
            "Arousal: % / 10 / relativ",
            y,
            12,
            12,
            8,
            "SELECT time, arousal_pct, arousal_10, arousal_rel * 100 AS arousal_rel_pct FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            unit="percent",
            min_v=0,
            max_v=100,
        )
    )
    y += 8

    panels.append(
        ts_panel(
            22,
            "Activation index",
            y,
            0,
            12,
            7,
            "SELECT time, activation_index FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    panels.append(
        ts_panel(
            23,
            "Z-score HR & RMSSD vs baseline",
            y,
            12,
            12,
            7,
            "SELECT time, z_hr, z_rmssd FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=-3,
            max_v=3,
        )
    )
    y += 7

    panels.append(row_panel("Calitate semnal & fiabilitate", y, 103))
    y += 1
    panels.append(
        ts_panel(
            30,
            "Incredere & trust",
            y,
            0,
            12,
            7,
            "SELECT time, confidence, epoch_trust, fresh_score FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=0,
            max_v=1,
        )
    )
    panels.append(
        ts_panel(
            31,
            "IBI count & fereastra HRV (s)",
            y,
            12,
            12,
            7,
            "SELECT time, ibi_n, window_sec FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    y += 7

    panels.append(
        timeline_panel(
            32,
            "Calitate date + mod scor",
            y,
            0,
            24,
            5,
            "SELECT time, data_quality, score_mode FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    y += 5

    panels.append(
        ts_panel(
            33,
            "Flag-uri fiabilitate (0/1)",
            y,
            0,
            24,
            6,
            "SELECT time, signal_trustworthy, motion_gated, context_suppress_alert, profile_ready, session_baseline_ready FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=0,
            max_v=1,
        )
    )
    y += 6

    panels.append(row_panel("Miscare & context", y, 104))
    y += 1
    panels.append(
        ts_panel(
            40,
            "Accelerometru: live / epoca / P90",
            y,
            0,
            12,
            8,
            "SELECT time, acc_rms AS acc_live FROM biofizic_acc_live WHERE $__timeFilter(time) AND acc_rms > 0 ORDER BY time",
            [
                "SELECT time, motion_acc_rms AS acc_epoch FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
                "SELECT time, acc_window_p90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
            unit="accMS2",
            min_v=0,
            max_v=5,
        )
    )
    panels.append(
        timeline_panel(
            41,
            "Mod activitate",
            y,
            12,
            12,
            8,
            "SELECT time, activity_mode FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    y += 8

    panels.append(row_panel("Baseline HR personal", y, 105))
    y += 1
    panels.append(
        ts_panel(
            50,
            "HR vs baseline lent + banda",
            y,
            0,
            24,
            8,
            "SELECT time, mean_hr, hr_baseline_slow, hr_band_lo, hr_band_hi FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            unit="bpm",
            min_v=45,
            max_v=140,
        )
    )
    y += 8

    panels.append(row_panel("Fusion v2 + v3", y, 106))
    y += 1
    panels.append(
        ts_panel(
            60,
            "Arousal v2 (10) vs fused",
            y,
            0,
            12,
            8,
            "SELECT time, arousal_10 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ["SELECT time, arousal_fused FROM biofizic_combined WHERE $__timeFilter(time) ORDER BY time"],
            min_v=0,
            max_v=10,
        )
    )
    panels.append(
        ts_panel(
            61,
            "Incredere v2 / v3 / fused",
            y,
            12,
            12,
            8,
            "SELECT time, confidence AS confidence_state FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            [
                "SELECT time, confidence_v2, confidence_v3, confidence_fused FROM biofizic_combined WHERE $__timeFilter(time) ORDER BY time",
            ],
            min_v=0,
            max_v=1,
        )
    )
    y += 8

    panels.append(
        timeline_panel(
            62,
            "Etichete v2 state + fusion",
            y,
            0,
            24,
            6,
            "SELECT time, emotion_v2, arousal_label FROM biofizic_combined WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    y += 6

    panels.append(row_panel("PPG (canal secundar)", y, 107))
    y += 1
    panels.append(
        ts_panel(
            70,
            "z_pulse_amp",
            y,
            0,
            12,
            7,
            "SELECT time, z_pulse_amp FROM biofizic_ppg_hrv WHERE $__timeFilter(time) ORDER BY time",
            min_v=-3,
            max_v=3,
        )
    )
    panels.append(
        ts_panel(
            71,
            "PPG: RMSSD + HR + peaks",
            y,
            12,
            12,
            7,
            "SELECT time, rmssd_ppg, mean_hr_ppg, peak_count FROM biofizic_ppg_hrv WHERE $__timeFilter(time) ORDER BY time",
        )
    )

    dashboard = {
        "uid": "biofizic-live",
        "title": "Biofizic — Monitor Live",
        "tags": ["biofizic", "live", "hrv", "vr"],
        "timezone": "Europe/Bucharest",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "10s",
        "time": {"from": "now-6h", "to": "now"},
        "graphTooltip": 1,
        "annotations": {
            "list": [
                {
                    "builtIn": 1,
                    "datasource": {"type": "grafana", "uid": "-- Grafana --"},
                    "enable": True,
                    "hide": True,
                    "iconColor": "rgba(0, 211, 255, 1)",
                    "name": "Annotations & Alerts",
                    "type": "dashboard",
                }
            ]
        },
        "panels": panels,
    }

    OUT.write_text(json.dumps(dashboard, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {OUT} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
