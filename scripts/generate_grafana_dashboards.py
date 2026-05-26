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
            "RMSSD multi-window (30/60/90 s)",
            4,
            "SELECT time, rmssd_w30, rmssd_w60, rmssd_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            unit="ms",
        ),
        ts_panel(
            6,
            "Stress index multi-window",
            12,
            "SELECT time, stress_index_w30, stress_index_w60, stress_index_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
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


def build_signal_quality_dashboard() -> dict:
    panels = [
        stat_panel(
            1,
            "Motion state",
            "SELECT motion_state AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0,
            0,
            w=6,
            decimals=0,
        ),
        stat_panel(
            2,
            "Signal quality (Q)",
            "SELECT signal_quality AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            w=6,
            decimals=2,
        ),
        stat_panel(
            3,
            "IBI artifact rate",
            "SELECT artifact_rate AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            w=6,
            decimals=3,
        ),
        timeline_panel(
            4,
            "Motion state over time",
            "SELECT time, motion_state FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            4,
            h=5,
        ),
        ts_panel(
            5,
            "Signal quality vs artifact rate",
            9,
            "SELECT time, signal_quality FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, artifact_rate FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
            h=7,
            min_v=0,
            max_v=1,
            decimals=3,
        ),
        ts_panel(
            6,
            "Cardiac-band motion energy vs acceleration (1 Hz batch)",
            16,
            "SELECT time, acc_band_cardiac FROM biofizic_acquisition_batch WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, acc_rms FROM biofizic_acquisition_batch WHERE $__timeFilter(time) ORDER BY time",
            ],
        ),
    ]
    return dashboard_meta(
        "biofizic-signal-quality",
        "Biofizic Signal Quality",
        ["biofizic", "quality"],
        panels,
        version=1,
    )


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


def build_stream_sync_dashboard() -> dict:
    """
    Diagnostic dashboard for the acquisition/batch v2 atomic sync.

    The thesis claims IBI and motion are bundled with a shared ts_anchor so the
    server can run HRV math on data from the same time window. These panels make
    that visible: anchor_delay_ms should stay positive (or zero when no fresher
    stream is available), seq should increment by 1 every second, the per-batch
    ibi_count trace shows the bursty nature of the HR stream that motivated the
    design, and acc_band_cardiac is the cardiac-band motion energy that feeds
    the signal-quality gate.
    """
    panels = [
        stat_panel(
            1,
            "Last anchor delay (ms)",
            "SELECT anchor_delay_ms AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0, 0, w=6, decimals=0,
        ),
        stat_panel(
            2,
            "Last seq",
            "SELECT seq AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6, 0, w=6, decimals=0,
        ),
        stat_panel(
            3,
            "Last IBI count",
            "SELECT ibi_count AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12, 0, w=6, decimals=0,
        ),
        stat_panel(
            4,
            "Last cardiac-band energy",
            "SELECT acc_band_cardiac AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18, 0, w=6, decimals=4,
        ),
        ts_panel(
            5,
            "Atomic anchor delay (ts_anchor - ts_publish) [ms]",
            4,
            "SELECT time, anchor_delay_ms FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            unit="ms",
            decimals=0,
        ),
        ts_panel(
            6,
            "IBI count per batch (Samsung HR bursts every ~4 s)",
            11,
            "SELECT time, ibi_count FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            min_v=0,
            decimals=0,
        ),
        ts_panel(
            7,
            "Cardiac-band motion energy per batch (0.5-4 Hz)",
            18,
            "SELECT time, acc_band_cardiac FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            min_v=0,
            decimals=4,
        ),
        ts_panel(
            8,
            "Skin temp age at publish [ms]",
            25,
            "SELECT time, skin_temp_age_ms FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) AND skin_temp_age_ms IS NOT NULL ORDER BY time",
            h=6,
            unit="ms",
            decimals=0,
        ),
        ts_panel(
            9,
            "Sequence number (should increment by 1 per second)",
            31,
            "SELECT time, seq FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=5,
            decimals=0,
        ),
    ]
    return dashboard_meta(
        "biofizic-stream-sync",
        "Biofizic Stream Sync Diagnostics",
        ["biofizic", "diagnostics", "atomic-sync"],
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
            "Motion state",
            "SELECT time, motion_state FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
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


def build_all_data_live_dashboard() -> dict:
    """High-frequency raw view: the PPG pulse wave with detected peaks, plus the
    IBI the watch reports vs the IBI reconstructed from those peaks."""
    panels = [
        ts_panel(
            1,
            "Raw PPG (green) with detected peaks",
            0,
            "SELECT time, ppg_green FROM biofizic_all_data_live "
            "WHERE $__timeFilter(time) AND ppg_green IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, ppg_peak FROM biofizic_all_data_live "
                "WHERE $__timeFilter(time) AND ppg_peak IS NOT NULL ORDER BY time",
            ],
            h=9,
        ),
        ts_panel(
            2,
            "IBI: watch (SDK) vs reconstructed from PPG peaks [ms]",
            9,
            "SELECT time, ibi_ms FROM biofizic_all_data_live "
            "WHERE $__timeFilter(time) AND ibi_ms IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, ibi_recon_mean FROM biofizic_legacy_ppg "
                "WHERE $__timeFilter(time) AND ibi_recon_mean > 0 ORDER BY time",
            ],
            unit="ms",
        ),
        stat_panel(
            3,
            "PPG sample rate (Hz)",
            "SELECT sample_rate_hz AS v FROM biofizic_legacy_ppg "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0, 17, w=12,
        ),
        stat_panel(
            4,
            "Peaks per window",
            "SELECT n_peaks AS v FROM biofizic_legacy_ppg "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12, 17, w=12, decimals=0,
        ),
    ]
    return dashboard_meta(
        "biofizic-all-data-live", "Biofizic ALL DATA LIVE", ["biofizic", "raw"], panels, refresh="1s"
    )


def build_determinist_vs_wesad_dashboard() -> dict:
    """Deterministic personal-baseline stress vs a WESAD-trained ML probability
    (a foreign-dataset model; expected to be noisier / more false-positive)."""
    panels = [
        ts_panel(
            1,
            "Filtered personal z (deterministic) vs WESAD P(stress)",
            0,
            "SELECT time, z_si_filtered FROM biofizic_state "
            "WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, p_stress FROM biofizic_legacy_wesad "
                "WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            2,
            "Production arousal (1-10) vs WESAD P(stress) x10",
            8,
            "SELECT time, arousal_10 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, p_stress * 10 AS wesad_x10 FROM biofizic_legacy_wesad "
                "WHERE $__timeFilter(time) ORDER BY time",
            ],
            min_v=0, max_v=10, decimals=0,
        ),
    ]
    return dashboard_meta(
        "biofizic-determinist-vs-wesad",
        "Biofizic Determinist vs WESAD",
        ["biofizic", "legacy"],
        panels,
    )


def build_ppg_failure_dashboard() -> dict:
    """PPG pulse amplitude collapses under wrist motion — the evidence for not
    estimating valence from wrist PPG."""
    panels = [
        ts_panel(
            1,
            "Pulse amplitude (PPA) vs wrist acceleration",
            0,
            "SELECT time, ppa FROM biofizic_legacy_ppg "
            "WHERE $__timeFilter(time) AND ppa > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, acc_rms FROM biofizic_acquisition_batch "
                "WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            2,
            "PPA z-score vs cardiac-band motion energy",
            8,
            "SELECT time, ppa_z FROM biofizic_legacy_ppg WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, acc_band_cardiac FROM biofizic_acquisition_batch "
                "WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-ppg-failure", "Biofizic PPG Failure in the Wild", ["biofizic", "legacy"], panels
    )


def build_valence_demo_dashboard() -> dict:
    """Ad-hoc valence heuristic (documented negative result): noisy and not
    separable from arousal; never used in production."""
    panels = [
        ts_panel(
            1,
            "Valence heuristic [-1,1] (legacy, NOT production)",
            0,
            "SELECT time, valence FROM biofizic_legacy_valence WHERE $__timeFilter(time) ORDER BY time",
            min_v=-1, max_v=1, decimals=2,
        ),
        ts_panel(
            2,
            "Valence inputs: RMSSD z vs PPA z",
            8,
            "SELECT time, rmssd_z FROM biofizic_legacy_valence WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, ppa_z FROM biofizic_legacy_valence WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-valence-demo", "Biofizic Valence Demo (negative result)", ["biofizic", "legacy"], panels
    )


def build_live_sync_dashboard() -> dict:
    """Validate that HR / RMSSD / motion are time-aligned: everything here comes
    from biofizic_live, all stamped with the watch ts_anchor (one clock)."""
    panels = [
        ts_panel(
            1,
            "Heart rate: SDK (instant) vs window mean_hr",
            0,
            "SELECT time, hr_sdk FROM biofizic_live WHERE $__timeFilter(time) AND hr_sdk > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, mean_hr FROM biofizic_live WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            ],
            unit="none",
        ),
        ts_panel(
            2,
            "HR (SDK) vs RMSSD — should move together (inverse) if aligned",
            8,
            "SELECT time, hr_sdk FROM biofizic_live WHERE $__timeFilter(time) AND hr_sdk > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd FROM biofizic_live WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
        ),
        ts_panel(
            3,
            "Motion (acc_rms / cardiac-band) on the same axis",
            16,
            "SELECT time, acc_rms FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, acc_band_cardiac FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=3,
        ),
    ]
    return dashboard_meta(
        "biofizic-live-sync", "Biofizic Live Sync", ["biofizic", "live"], panels, refresh="1s"
    )


def build_reliability_dashboard() -> dict:
    """How trustworthy each verdict is (VR context): arousal vs quality, the
    fusion weight, the HRV/HR z channels, and the Kalman gain."""
    panels = [
        ts_panel(
            1,
            "Arousal (1-10) vs confidence Q",
            0,
            "SELECT time, arousal_10 FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, signal_quality FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            2,
            "Fusion: z_hrv vs z_hr vs z_filtered (hrv_weight shows the blend)",
            8,
            "SELECT time, z_hrv FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, z_hr FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
                "SELECT time, z_filtered FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
                "SELECT time, hrv_weight FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            3,
            "Kalman gain & artifact rate (low gain / high artifacts => held verdict)",
            16,
            "SELECT time, kalman_gain FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, artifact_rate FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=3,
        ),
    ]
    return dashboard_meta(
        "biofizic-reliability", "Biofizic Reliability", ["biofizic", "live"], panels, refresh="1s"
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    dashboards = {
        "biofizic-hrv-analysis.json": build_hrv_dashboard(),
        "biofizic-baseline-compare.json": build_baseline_dashboard(),
        "biofizic-signal-quality.json": build_signal_quality_dashboard(),
        "biofizic-live-overview.json": build_overview_dashboard(),
        "biofizic-window-comparison.json": build_window_comparison_dashboard(),
        "biofizic-stream-sync.json": build_stream_sync_dashboard(),
        "biofizic-session-overview.json": build_session_overview_dashboard(),
        "biofizic-all-data-live.json": build_all_data_live_dashboard(),
        "biofizic-live-sync.json": build_live_sync_dashboard(),
        "biofizic-reliability.json": build_reliability_dashboard(),
        "biofizic-determinist-vs-wesad.json": build_determinist_vs_wesad_dashboard(),
        "biofizic-ppg-failure.json": build_ppg_failure_dashboard(),
        "biofizic-valence-demo.json": build_valence_demo_dashboard(),
    }
    for name, body in dashboards.items():
        path = OUT / name
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
