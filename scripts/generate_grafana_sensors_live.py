#!/usr/bin/env python3
"""Genereaza dashboard Grafana 1 Hz — senzori ceas + clasificare + corelatii."""

import json
from pathlib import Path

DS = {"type": "influxdb", "uid": "biofizic-influx"}
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docker/grafana/provisioning/dashboards/biofizic-sensors-live.json"


def ds(sql: str, ref: str = "A", fmt: str = "time_series") -> dict:
    return {
        "datasource": DS,
        "editorMode": "code",
        "format": fmt,
        "rawQuery": True,
        "rawSql": sql,
        "refId": ref,
    }


def row(title: str, y: int, rid: int) -> dict:
    return {
        "id": rid,
        "type": "row",
        "title": title,
        "gridPos": {"h": 1, "w": 24, "x": 0, "y": y},
        "collapsed": False,
        "panels": [],
    }


def ts(
    pid: int,
    title: str,
    y: int,
    x: int,
    w: int,
    h: int,
    sql: str,
    extra: list[str] | None = None,
    unit: str | None = None,
    min_v: float | None = None,
    max_v: float | None = None,
    points: str = "never",
) -> dict:
    targets = [ds(sql)]
    for i, s in enumerate(extra or [], 1):
        targets.append(ds(s, ref=chr(ord("A") + i)))
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
                    "fillOpacity": 8,
                    "showPoints": points,
                    "spanNulls": True,
                    "lineInterpolation": "linear",
                },
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


def stat(pid: int, title: str, y: int, x: int, w: int, sql: str, unit: str | None = None) -> dict:
    defaults: dict = {
        "color": {"mode": "thresholds"},
        "thresholds": {"mode": "absolute", "steps": [{"color": "green", "value": None}]},
    }
    if unit:
        defaults["unit"] = unit
    return {
        "id": pid,
        "type": "stat",
        "title": title,
        "gridPos": {"h": 4, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds(sql, fmt="table")],
        "fieldConfig": {"defaults": defaults, "overrides": []},
        "options": {
            "reduceOptions": {"calcs": ["lastNotNull"], "fields": "", "values": False},
            "colorMode": "value",
            "graphMode": "area",
            "textMode": "auto",
        },
    }


def timeline(pid: int, title: str, y: int, x: int, w: int, h: int, sql: str) -> dict:
    return {
        "id": pid,
        "type": "state-timeline",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds(sql, fmt="table")],
        "fieldConfig": {"defaults": {"custom": {"fillOpacity": 80}}, "overrides": []},
        "options": {
            "mergeValues": True,
            "showValue": "auto",
            "legend": {"displayMode": "list", "placement": "bottom", "showLegend": True},
        },
    }


def xy(pid: int, title: str, y: int, x: int, w: int, h: int, sql: str) -> dict:
    return {
        "id": pid,
        "type": "xychart",
        "title": title,
        "gridPos": {"h": h, "w": w, "x": x, "y": y},
        "datasource": DS,
        "targets": [ds(sql, fmt="table")],
        "options": {
            "legend": {"showLegend": True},
            "tooltip": {"mode": "single"},
        },
    }


def main() -> None:
    W = "biofizic_watch_live"
    panels: list[dict] = []
    y = 0
    pid = 1

    panels.append(row("Rezumat LIVE (1 Hz)", y, 900))
    y += 1
    stats = [
        ("HR", f"SELECT hr AS v FROM {W} WHERE $__timeFilter(time) AND hr > 0 ORDER BY time DESC LIMIT 1", "bpm"),
        ("RMSSD live", f"SELECT rmssd_live AS v FROM {W} WHERE $__timeFilter(time) AND rmssd_live > 0 ORDER BY time DESC LIMIT 1", "ms"),
        ("PPG n/s", f"SELECT ppg_n AS v FROM {W} WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
        ("Arousal", f"SELECT server_arousal_10 AS v FROM {W} WHERE $__timeFilter(time) AND server_arousal_10 >= 0 ORDER BY time DESC LIMIT 1", None),
        ("Emoție", f"SELECT server_emotion AS v FROM {W} WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
        ("Mod", f"SELECT server_activity_mode AS v FROM {W} WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1", None),
    ]
    for i, (title, sql, unit) in enumerate(stats):
        panels.append(stat(pid, title, y, i * 4, 4, sql, unit))
        pid += 1
    y += 4

    panels.append(row("Cardiac — individual 1 Hz", y, 901))
    y += 1
    panels.append(
        ts(
            pid,
            "HR (ceas)",
            y,
            0,
            12,
            7,
            f"SELECT time, hr FROM {W} WHERE $__timeFilter(time) AND hr > 0 ORDER BY time",
            [
                f"SELECT time, mean_hr_live AS hr_ibi FROM {W} WHERE $__timeFilter(time) AND mean_hr_live > 0 ORDER BY time",
                "SELECT time, hr AS hr_server FROM biofizic_state_live WHERE $__timeFilter(time) AND hr > 0 ORDER BY time",
            ],
            unit="bpm",
            min_v=45,
            max_v=150,
            points="auto",
        )
    )
    pid += 1
    panels.append(
        ts(
            pid,
            "IBI buffer & fereastră HRV",
            y,
            12,
            12,
            7,
            f"SELECT time, ibi_n FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            [f"SELECT time, ibi_window_sec FROM {W} WHERE $__timeFilter(time) ORDER BY time"],
            unit="short",
        )
    )
    pid += 1
    y += 7

    panels.append(row("HRV — rolling ceas vs epoch vs PPG", y, 902))
    y += 1
    panels.append(
        ts(
            pid,
            "RMSSD: live 1s vs epoch 30s vs PPG processor",
            y,
            0,
            24,
            8,
            f"SELECT time, rmssd_live AS rmssd_watch FROM {W} WHERE $__timeFilter(time) AND rmssd_live > 0 ORDER BY time",
            [
                "SELECT time, rmssd AS rmssd_epoch FROM biofizic_epoch WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
                "SELECT time, rmssd_ppg FROM biofizic_ppg_hrv WHERE $__timeFilter(time) AND rmssd_ppg > 0 ORDER BY time",
                "SELECT time, rmssd AS rmssd_state FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            unit="ms",
            min_v=0,
            max_v=200,
        )
    )
    pid += 1
    y += 8
    panels.append(
        ts(
            pid,
            "SDNN & pNN50 (ceas rolling)",
            y,
            0,
            12,
            6,
            f"SELECT time, sdnn_live FROM {W} WHERE $__timeFilter(time) AND sdnn_live > 0 ORDER BY time",
            [f"SELECT time, pnn50_live FROM {W} WHERE $__timeFilter(time) ORDER BY time"],
            unit="ms",
        )
    )
    pid += 1
    panels.append(
        xy(
            pid,
            "Corelație HR vs RMSSD (ceas, 1 Hz)",
            y,
            12,
            12,
            6,
            f"SELECT hr AS x, rmssd_live AS y FROM {W} WHERE $__timeFilter(time) AND hr > 0 AND rmssd_live > 0 ORDER BY time",
        )
    )
    pid += 1
    y += 6

    panels.append(row("Mișcare — ACC & Gyro 1 Hz", y, 903))
    y += 1
    panels.append(
        ts(
            pid,
            "ACC RMS & magnitudine",
            y,
            0,
            12,
            6,
            f"SELECT time, acc_rms FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            [f"SELECT time, acc_mag FROM {W} WHERE $__timeFilter(time) ORDER BY time"],
            unit="accMS2",
        )
    )
    pid += 1
    panels.append(
        ts(
            pid,
            "Gyro RMS",
            y,
            12,
            12,
            6,
            f"SELECT time, gyro_rms FROM {W} WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    pid += 1
    y += 6
    panels.append(
        ts(
            pid,
            "Context server: motion_z & activity",
            y,
            0,
            24,
            6,
            f"SELECT time, server_motion_z FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            [
                "SELECT time, motion_z FROM biofizic_context_live WHERE $__timeFilter(time) ORDER BY time",
                f"SELECT time, server_z_hr FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            ],
        )
    )
    pid += 1
    y += 6

    panels.append(row("PPG raw — individual 1 Hz (medii/sec)", y, 904))
    y += 1
    for title, col in [
        ("PPG GREEN mean ± std", "ppg_green_mean"),
        ("PPG IR mean", "ppg_ir_mean"),
        ("PPG RED mean", "ppg_red_mean"),
    ]:
        extra = None
        if col == "ppg_green_mean":
            extra = [f"SELECT time, ppg_green_std FROM {W} WHERE $__timeFilter(time) ORDER BY time"]
        panels.append(
            ts(
                pid,
                title,
                y,
                0 if col == "ppg_green_mean" else (8 if col == "ppg_ir_mean" else 16),
                8 if col != "ppg_green_mean" else 24,
                6 if col == "ppg_green_mean" else 6,
                f"SELECT time, {col} FROM {W} WHERE $__timeFilter(time) AND {col} > 0 ORDER BY time",
                extra if col == "ppg_green_mean" else None,
            )
        )
        if col == "ppg_green_mean":
            y += 6
        pid += 1
    panels.append(
        ts(
            pid,
            "PPG eșantioane / secundă",
            y,
            0,
            8,
            5,
            f"SELECT time, ppg_n FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            points="auto",
        )
    )
    pid += 1
    y += 5

    panels.append(row("PPG procesat (server) — comparativ", y, 905))
    y += 1
    ppg_metrics = [
        ("RMSSD PPG", "rmssd_ppg", "ms"),
        ("SDNN PPG", "sdnn_ppg", "ms"),
        ("Pulse amp mean", "pulse_amp_mean", None),
        ("z_pulse_amp (simpatic)", "z_pulse_amp", None),
        ("Peak count", "peak_count", None),
        ("Mean HR PPG", "mean_hr_ppg", "bpm"),
    ]
    for i, (title, field, unit) in enumerate(ppg_metrics):
        panels.append(
            ts(
                pid,
                title,
                y + (i // 2) * 6,
                (i % 2) * 12,
                12,
                6,
                f"SELECT time, {field} FROM biofizic_ppg_hrv WHERE $__timeFilter(time) AND {field} IS NOT NULL ORDER BY time",
                unit=unit,
            )
        )
        pid += 1
    y += 18

    panels.append(row("Temperatură & pași", y, 906))
    y += 1
    panels.append(
        ts(
            pid,
            "Temperatură piele & ambient",
            y,
            0,
            12,
            6,
            f"SELECT time, skin_temp_c FROM {W} WHERE $__timeFilter(time) AND skin_temp_c > 20 ORDER BY time",
            [f"SELECT time, ambient_temp_c FROM {W} WHERE $__timeFilter(time) AND ambient_temp_c > 0 ORDER BY time"],
            unit="celsius",
        )
    )
    pid += 1
    panels.append(
        ts(
            pid,
            "Pași (cumulativ)",
            y,
            12,
            12,
            6,
            f"SELECT time, steps FROM {W} WHERE $__timeFilter(time) ORDER BY time",
        )
    )
    pid += 1
    y += 6

    panels.append(row("Clasificare LIVE — la fiecare secundă", y, 907))
    y += 1
    panels.append(
        ts(
            pid,
            "Arousal /10: server live vs fusion vs ceas",
            y,
            0,
            24,
            7,
            f"SELECT time, server_arousal_10 AS arousal_server FROM {W} WHERE $__timeFilter(time) AND server_arousal_10 >= 0 ORDER BY time",
            [
                "SELECT time, arousal_10 AS arousal_state FROM biofizic_state_live WHERE $__timeFilter(time) ORDER BY time",
                "SELECT time, arousal_fused FROM biofizic_combined WHERE $__timeFilter(time) ORDER BY time",
                f"SELECT time, arousal_10 AS arousal_watch FROM {W} WHERE $__timeFilter(time) AND arousal_10 >= 0 ORDER BY time",
            ],
            min_v=0,
            max_v=10,
            points="auto",
        )
    )
    pid += 1
    y += 7
    panels.append(
        timeline(
            pid,
            "Emoție & mod activitate (timeline)",
            y,
            0,
            24,
            5,
            f"""SELECT time, server_emotion AS emotion FROM {W} WHERE $__timeFilter(time) AND server_emotion IS NOT NULL ORDER BY time""",
        )
    )
    pid += 1
    y += 5
    panels.append(
        ts(
            pid,
            "Confidence & stress index",
            y,
            0,
            12,
            6,
            f"SELECT time, confidence FROM {W} WHERE $__timeFilter(time) ORDER BY time",
            [
                "SELECT time, confidence_fused FROM biofizic_combined WHERE $__timeFilter(time) ORDER BY time",
                "SELECT time, stress_index FROM biofizic_state WHERE $__timeFilter(time) AND stress_index > 0 ORDER BY time",
            ],
            min_v=0,
            max_v=1,
        )
    )
    pid += 1
    panels.append(
        ts(
            pid,
            "Baevsky / activation (epoch 30s)",
            y,
            12,
            12,
            6,
            "SELECT time, stress_index FROM biofizic_state WHERE $__timeFilter(time) AND stress_index > 0 ORDER BY time",
            [
                "SELECT time, baevsky_si FROM biofizic_state WHERE $__timeFilter(time) AND baevsky_si > 0 ORDER BY time",
                "SELECT time, activation_index FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
        )
    )
    pid += 1
    y += 6

    panels.append(row("Calitate semnal", y, 908))
    y += 1
    panels.append(
        ts(
            pid,
            "sec_since_ibi & hrv_ready",
            y,
            0,
            24,
            5,
            f"SELECT time, sec_since_ibi FROM {W} WHERE $__timeFilter(time) AND sec_since_ibi >= 0 ORDER BY time",
            points="auto",
        )
    )
    pid += 1

    dashboard = {
        "uid": "biofizic-sensors-live",
        "title": "Biofizic — Senzori LIVE 1 Hz",
        "tags": ["biofizic", "live", "sensors", "1hz"],
        "timezone": "browser",
        "schemaVersion": 39,
        "version": 1,
        "refresh": "1s",
        "time": {"from": "now-15m", "to": "now"},
        "fiscalYearStartMonth": 0,
        "liveNow": True,
        "panels": panels,
        "templating": {"list": []},
        "annotations": {"list": []},
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(dashboard, indent=2), encoding="utf-8")
    print(f"Wrote {OUT} ({len(panels)} panels)")


if __name__ == "__main__":
    main()
