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
            "RMSSD 30s (now) — higher = calmer",
            "SELECT rmssd AS v FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time DESC LIMIT 1",
            0,
            0,
            unit="ms",
        ),
        stat_panel(
            2,
            "Stress index (now) — higher = more stress",
            "SELECT stress_index AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            decimals=2,
        ),
        stat_panel(
            3,
            "Label: Kubios (population)",
            "SELECT emotion AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            decimals=0,
        ),
        stat_panel(
            4,
            "Label: personal baseline",
            "SELECT emotion_baseline AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18,
            0,
            decimals=0,
        ),
        ts_panel(
            5,
            "RMSSD 30/60/90s — short reacts fast, long is stable",
            4,
            "SELECT time, rmssd_w30, rmssd_w60, rmssd_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            unit="ms",
        ),
        ts_panel(
            6,
            "Stress index 30/60/90s — compare window responsiveness",
            12,
            "SELECT time, stress_index_w30, stress_index_w60, stress_index_w90 FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            decimals=2,
        ),
        ts_panel(
            7,
            "HR vs RMSSD — should move inversely if signal is valid",
            20,
            "SELECT time, mean_hr AS hr_bpm FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd AS rmssd_ms FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            unit="none",
        ),
    ]
    return dashboard_meta("biofizic-hrv-analysis", "Biofizic HRV Analysis", ["biofizic", "hrv"], panels, version=4)


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
            "Kubios vs personal labels — watch where they disagree",
            "SELECT time, emotion, emotion_baseline FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            0,
        ),
        ts_panel(
            2,
            "Label levels 1-5 + agreement — gap = population vs personal differ",
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
            "Stress index now vs your resting baseline",
            13,
            "SELECT time, stress_index AS si_kubios FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, baseline_si AS si_baseline FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            4,
            "Stress z-score — >0 above your rest, <0 below",
            21,
            "SELECT time, z_si FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=-3,
            max_v=3,
            decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-baseline-compare", "Biofizic Baseline Compare", ["biofizic", "baseline"], panels, version=4
    )


def build_signal_quality_dashboard() -> dict:
    panels = [
        stat_panel(
            1,
            "Motion state (now)",
            "SELECT motion_state AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0,
            0,
            w=6,
            decimals=0,
        ),
        stat_panel(
            2,
            "Signal quality Q (now) — 1=clean, 0=unusable",
            "SELECT signal_quality AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            w=6,
            decimals=2,
        ),
        stat_panel(
            3,
            "IBI artifact rate (now) — lower is better",
            "SELECT artifact_rate AS v FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            w=6,
            decimals=3,
        ),
        timeline_panel(
            4,
            "Motion timeline — still vs moving",
            "SELECT time, motion_state FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            4,
            h=5,
        ),
        ts_panel(
            5,
            "Q vs artifact rate — Q drops as artifacts rise",
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
            "Cardiac-band motion vs total accel — the part that corrupts PPG",
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
        version=2,
    )


def build_overview_dashboard() -> dict:
    """Live overview from biofizic_state_live (1 Hz). Epoch decisions remain on biofizic_state."""
    panels = [
        stat_panel(
            1,
            "Arousal now (1-10)",
            "SELECT arousal_10 AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time DESC LIMIT 1",
            0,
            0,
            decimals=0,
        ),
        stat_panel(
            2,
            "Kubios label (now)",
            "SELECT emotion AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6,
            0,
            decimals=0,
        ),
        stat_panel(
            3,
            "IBI buffer — beats available for HRV",
            "SELECT ibi_buffer_size AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12,
            0,
            decimals=0,
        ),
        stat_panel(
            4,
            "Data quality (now)",
            "SELECT data_quality AS v FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18,
            0,
            decimals=0,
        ),
        ts_panel(
            5,
            "Arousal 1Hz — live activation trend",
            4,
            "SELECT time, arousal_10 FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time",
            min_v=0,
            max_v=10,
            decimals=0,
        ),
        ts_panel(
            6,
            "RMSSD & stress index (live) — the markers behind arousal",
            12,
            "SELECT time, rmssd, stress_index FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            decimals=2,
        ),
        timeline_panel(
            7,
            "Data quality timeline",
            "SELECT time, data_quality FROM biofizic_state_live "
            "WHERE $__timeFilter(time) ORDER BY time",
            20,
            h=4,
        ),
        ts_panel(
            8,
            "Skin temp & HR (1Hz raw)",
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
        version=7,
        refresh="1s",
    )


def build_window_comparison_dashboard() -> dict:
    """Compare HRV metrics across 30/60/90s windows from biofizic_state_windows."""
    panels = [
        ts_panel(
            1,
            "RMSSD per window — convergence = stable HRV",
            0,
            "SELECT time, w30_rmssd, w60_rmssd, w90_rmssd FROM biofizic_state_windows "
            "WHERE $__timeFilter(time) AND w30_rmssd > 0 ORDER BY time",
            unit="ms",
        ),
        ts_panel(
            2,
            "Stress index per window — short vs long agreement",
            8,
            "SELECT time, w30_stress_index, w60_stress_index, w90_stress_index "
            "FROM biofizic_state_windows WHERE $__timeFilter(time) ORDER BY time",
            decimals=2,
        ),
        ts_panel(
            3,
            "IBI count per window — enough beats to trust it?",
            16,
            "SELECT time, w30_ibi_count, w60_ibi_count, w90_ibi_count "
            "FROM biofizic_state_windows WHERE $__timeFilter(time) ORDER BY time",
            decimals=0,
        ),
        stat_panel(
            4,
            "IBI buffer size (now)",
            "SELECT ibi_buffer_size AS v FROM biofizic_state_windows "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0,
            24,
            w=8,
            decimals=0,
        ),
        timeline_panel(
            5,
            "Window quality timeline (30/60/90s)",
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
        version=2,
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
            "Anchor delay (now, ms) — should be >=0",
            "SELECT anchor_delay_ms AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            0, 0, w=6, decimals=0,
        ),
        stat_panel(
            2,
            "Seq (now) — should increment by 1/s",
            "SELECT seq AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            6, 0, w=6, decimals=0,
        ),
        stat_panel(
            3,
            "IBI count (last batch)",
            "SELECT ibi_count AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            12, 0, w=6, decimals=0,
        ),
        stat_panel(
            4,
            "Cardiac-band energy (now)",
            "SELECT acc_band_cardiac AS v FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time DESC LIMIT 1",
            18, 0, w=6, decimals=4,
        ),
        ts_panel(
            5,
            "Anchor delay over time — stays >=0 if batch is atomic",
            4,
            "SELECT time, anchor_delay_ms FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            unit="ms",
            decimals=0,
        ),
        ts_panel(
            6,
            "IBI per batch — bursty Samsung HR (~every 4s)",
            11,
            "SELECT time, ibi_count FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            min_v=0,
            decimals=0,
        ),
        ts_panel(
            7,
            "Cardiac-band motion per batch (0.5-4 Hz)",
            18,
            "SELECT time, acc_band_cardiac FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            min_v=0,
            decimals=4,
        ),
        ts_panel(
            8,
            "Skin temp age at publish (ms) — sensor freshness",
            25,
            "SELECT time, skin_temp_age_ms FROM biofizic_acquisition_batch "
            "WHERE $__timeFilter(time) AND skin_temp_age_ms IS NOT NULL ORDER BY time",
            h=6,
            unit="ms",
            decimals=0,
        ),
        ts_panel(
            9,
            "Seq over time — +1 per second = no dropped batches",
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
        version=2,
        refresh="5s",
    )


def build_session_overview_dashboard() -> dict:
    panels = [
        ts_panel(
            1,
            "HR & RMSSD — should move inversely if valid",
            0,
            "SELECT time, mean_hr AS hr_bpm FROM biofizic_state WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd AS rmssd_ms FROM biofizic_state WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            h=7,
        ),
        ts_panel(
            2,
            "Stress index & z-score — raw vs vs-your-baseline",
            7,
            "SELECT time, stress_index, z_si FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            h=7,
            decimals=2,
        ),
        timeline_panel(
            3,
            "Motion timeline — still vs moving",
            "SELECT time, motion_state FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            14,
            h=4,
        ),
        ts_panel(
            4,
            "Acceleration RMS — how much you moved",
            18,
            "SELECT time, acc_rms FROM biofizic_acquisition_batch WHERE $__timeFilter(time) ORDER BY time",
            h=6,
            unit="accMS2",
        ),
        ts_panel(
            5,
            "Arousal (1-10) over the session",
            24,
            "SELECT time, arousal_10 FROM biofizic_state "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time",
            min_v=0,
            max_v=10,
            decimals=0,
        ),
        ts_panel(
            6,
            "Label agreement — 5=Kubios & personal agree, 0=differ",
            32,
            "SELECT time, labels_agree * 5 AS labels_agree_scaled FROM biofizic_state WHERE $__timeFilter(time) ORDER BY time",
            min_v=0,
            max_v=5,
            decimals=0,
            h=5,
        ),
    ]
    return dashboard_meta(
        "biofizic-session-overview", "Biofizic Session Overview", ["biofizic", "session"], panels, version=4
    )


def build_all_data_live_dashboard() -> dict:
    """High-frequency raw view: the PPG pulse wave with detected peaks, plus the
    IBI the watch reports vs the IBI reconstructed from those peaks."""
    panels = [
        ts_panel(
            1,
            "Raw PPG (green) + detected peaks — clean pulse wave?",
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
            "IBI: watch SDK vs reconstructed from PPG peaks — should match",
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
        "biofizic-all-data-live", "Biofizic ALL DATA LIVE", ["biofizic", "raw"], panels, version=2, refresh="1s"
    )


def build_determinist_vs_wesad_dashboard() -> dict:
    """Deterministic personal-baseline stress vs a WESAD-trained ML probability
    (a foreign-dataset model; expected to be noisier / more false-positive)."""
    panels = [
        ts_panel(
            1,
            "Personal z (ours) vs WESAD P(stress) — ours is steadier",
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
            "Our arousal vs WESAD P(stress)x10 — WESAD over-flags on wrist",
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
        version=2,
    )


def build_ppg_failure_dashboard() -> dict:
    """PPG pulse amplitude collapses under wrist motion — the evidence for not
    estimating valence from wrist PPG."""
    panels = [
        ts_panel(
            1,
            "Pulse amplitude (PPA) vs accel — PPA collapses when you move",
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
            "PPA z-score vs cardiac-band motion — inverse under motion",
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
        "biofizic-ppg-failure", "Biofizic PPG Failure in the Wild", ["biofizic", "legacy"], panels, version=2
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
            "Valence inputs: RMSSD z vs PPA z — noisy & inseparable",
            8,
            "SELECT time, rmssd_z FROM biofizic_legacy_valence WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, ppa_z FROM biofizic_legacy_valence WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-valence-demo", "Biofizic Valence Demo (negative result)", ["biofizic", "legacy"], panels, version=2
    )


def build_live_sync_dashboard() -> dict:
    """Validate that HR / RMSSD / motion are time-aligned: everything here comes
    from biofizic_live, all stamped with the watch ts_anchor (one clock)."""
    panels = [
        ts_panel(
            1,
            "HR: SDK instant vs window mean — same trend if aligned",
            0,
            "SELECT time, hr_sdk FROM biofizic_live WHERE $__timeFilter(time) AND hr_sdk > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, mean_hr FROM biofizic_live WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            ],
            unit="none",
        ),
        ts_panel(
            2,
            "HR vs RMSSD — inverse & time-aligned if sync is correct",
            8,
            "SELECT time, hr_sdk FROM biofizic_live WHERE $__timeFilter(time) AND hr_sdk > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd FROM biofizic_live WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
        ),
        ts_panel(
            3,
            "Motion (acc_rms / cardiac-band) on the same time axis",
            16,
            "SELECT time, acc_rms FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, acc_band_cardiac FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=3,
        ),
    ]
    return dashboard_meta(
        "biofizic-live-sync", "Biofizic Live Sync", ["biofizic", "live"], panels, version=2, refresh="1s"
    )


def build_reliability_dashboard() -> dict:
    """How trustworthy each verdict is (VR context): arousal vs quality, the
    fusion weight, the HRV/HR z channels, and the Kalman gain."""
    panels = [
        ts_panel(
            1,
            "Arousal vs confidence Q — is the verdict trustworthy now?",
            0,
            "SELECT time, arousal_10 FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, signal_quality FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            2,
            "Fusion: z_hrv vs z_hr vs z_filtered (+weight) — which channel drives it",
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
            "Kalman gain & artifacts — low gain = verdict held (ignores noise)",
            16,
            "SELECT time, kalman_gain FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            extra_sql=[
                "SELECT time, artifact_rate FROM biofizic_live WHERE $__timeFilter(time) ORDER BY time",
            ],
            decimals=3,
        ),
    ]
    return dashboard_meta(
        "biofizic-reliability", "Biofizic Reliability", ["biofizic", "live"], panels, version=2, refresh="1s"
    )


def build_cardiac_comparator_dashboard() -> dict:
    """Three independent cardiac sources, same DSP, overlaid: which gives the
    most stable HR / RMSSD? PPG on-demand @100 Hz vs PPG continuous @25 Hz vs
    Samsung HR continuous IBI. The fourth series on each overlay is the
    production engine for reference, so we can also see how much the choice of
    source would shift the verdict the watch shows today."""
    panels = [
        # 1) Raw waveforms overlaid — same physical signal, two sample rates.
        ts_panel(
            1,
            "PPG green: on-demand (100 Hz) vs continuous (25 Hz) — same wave?",
            0,
            "SELECT time, green AS green_ondemand FROM biofizic_test_ppg_ondemand "
            "WHERE $__timeFilter(time) AND green IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, green AS green_continuous FROM biofizic_test_ppg_continuous "
                "WHERE $__timeFilter(time) AND green IS NOT NULL ORDER BY time",
            ],
            h=9, decimals=0,
        ),
        # 2) HR overlay — 3 derived sources + production for reference.
        ts_panel(
            2,
            "HR (bpm): ppg_ondemand vs ppg_continuous vs hr_continuous vs production",
            9,
            "SELECT time, hr_bpm AS hr_ondemand FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_ondemand' AND hr_bpm > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, hr_bpm AS hr_continuous FROM biofizic_test_derived "
                "WHERE $__timeFilter(time) AND source = 'ppg_continuous' AND hr_bpm > 0 ORDER BY time",
                "SELECT time, hr_bpm AS hr_hrtracker FROM biofizic_test_derived "
                "WHERE $__timeFilter(time) AND source = 'hr_continuous' AND hr_bpm > 0 ORDER BY time",
                "SELECT time, mean_hr AS hr_production FROM biofizic_state_live "
                "WHERE $__timeFilter(time) AND mean_hr > 0 ORDER BY time",
            ],
            unit="none",
        ),
        # 3) RMSSD overlay — the metric that actually drives the stress verdict.
        ts_panel(
            3,
            "RMSSD (ms): same three sources + production — accuracy thesis answer",
            18,
            "SELECT time, rmssd_ms AS rmssd_ondemand FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_ondemand' AND rmssd_ms > 0 ORDER BY time",
            extra_sql=[
                "SELECT time, rmssd_ms AS rmssd_continuous FROM biofizic_test_derived "
                "WHERE $__timeFilter(time) AND source = 'ppg_continuous' AND rmssd_ms > 0 ORDER BY time",
                "SELECT time, rmssd_ms AS rmssd_hrtracker FROM biofizic_test_derived "
                "WHERE $__timeFilter(time) AND source = 'hr_continuous' AND rmssd_ms > 0 ORDER BY time",
                "SELECT time, rmssd AS rmssd_production FROM biofizic_state_live "
                "WHERE $__timeFilter(time) AND rmssd > 0 ORDER BY time",
            ],
            unit="ms",
        ),
        # 4–6) Mean HR per source over the dashboard range (eyeball comparison).
        stat_panel(
            4, "Mean HR — ppg_ondemand",
            "SELECT AVG(hr_bpm) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_ondemand' AND hr_bpm > 0",
            0, 27, w=8, unit="none",
        ),
        stat_panel(
            5, "Mean HR — ppg_continuous",
            "SELECT AVG(hr_bpm) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_continuous' AND hr_bpm > 0",
            8, 27, w=8, unit="none",
        ),
        stat_panel(
            6, "Mean HR — hr_continuous (SDK)",
            "SELECT AVG(hr_bpm) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'hr_continuous' AND hr_bpm > 0",
            16, 27, w=8, unit="none",
        ),
        # 7–9) Mean RMSSD per source — divergence here is the metric that matters.
        stat_panel(
            7, "Mean RMSSD — ppg_ondemand",
            "SELECT AVG(rmssd_ms) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_ondemand' AND rmssd_ms > 0",
            0, 31, w=8, unit="ms",
        ),
        stat_panel(
            8, "Mean RMSSD — ppg_continuous",
            "SELECT AVG(rmssd_ms) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_continuous' AND rmssd_ms > 0",
            8, 31, w=8, unit="ms",
        ),
        stat_panel(
            9, "Mean RMSSD — hr_continuous (SDK)",
            "SELECT AVG(rmssd_ms) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'hr_continuous' AND rmssd_ms > 0",
            16, 31, w=8, unit="ms",
        ),
        # 10–12) Sample rate + artifact rate per source for context (was the
        # signal even clean enough to trust the derived values above?).
        stat_panel(
            10, "Sample rate Hz (ppg_ondemand)",
            "SELECT sample_rate_hz AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_ondemand' ORDER BY time DESC LIMIT 1",
            0, 35, w=8, unit="none", decimals=0,
        ),
        stat_panel(
            11, "Sample rate Hz (ppg_continuous)",
            "SELECT sample_rate_hz AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time) AND source = 'ppg_continuous' ORDER BY time DESC LIMIT 1",
            8, 35, w=8, unit="none", decimals=0,
        ),
        stat_panel(
            12, "Artifact rate (avg over range, all sources)",
            "SELECT AVG(artifact_rate) AS v FROM biofizic_test_derived "
            "WHERE $__timeFilter(time)",
            16, 35, w=8, unit="percentunit", decimals=2,
        ),
        # 13–15) Full PPG-only PhysiologyPipeline vs production (state/live).
        # This is the methodological core: same decision stack, only the IBI
        # source differs. Overlay shows whether the verdict would shift if
        # production used PPG-derived IBI instead of Samsung's processed IBI.
        ts_panel(
            13,
            "Arousal 1..10: production vs ppg_only_ondemand vs ppg_only_continuous",
            39,
            "SELECT time, arousal_10 AS arousal_production FROM biofizic_state_live "
            "WHERE $__timeFilter(time) AND arousal_10 IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, arousal_10 AS arousal_ppg_ondemand "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_ondemand' "
                "AND arousal_10 IS NOT NULL ORDER BY time",
                "SELECT time, arousal_10 AS arousal_ppg_continuous "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_continuous' "
                "AND arousal_10 IS NOT NULL ORDER BY time",
            ],
            min_v=0, max_v=10, decimals=0, h=9,
        ),
        # NOTE: z_si_filtered and confidence are NOT in biofizic_state_live
        # (only z_si is). The Kalman-filtered z and the multi-channel
        # confidence live on biofizic/live → measurement biofizic_live,
        # field name z_filtered (not z_si_filtered).
        ts_panel(
            14,
            "z (Kalman-filtered): same Kalman, different IBI input",
            48,
            "SELECT time, z_filtered AS z_production FROM biofizic_live "
            "WHERE $__timeFilter(time) AND z_filtered IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, z_si_filtered AS z_ppg_ondemand "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_ondemand' "
                "AND z_si_filtered IS NOT NULL ORDER BY time",
                "SELECT time, z_si_filtered AS z_ppg_continuous "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_continuous' "
                "AND z_si_filtered IS NOT NULL ORDER BY time",
            ],
            decimals=2,
        ),
        ts_panel(
            15,
            "Confidence + signal quality: how much each pipeline trusts itself",
            56,
            "SELECT time, confidence AS conf_production FROM biofizic_live "
            "WHERE $__timeFilter(time) AND confidence IS NOT NULL ORDER BY time",
            extra_sql=[
                "SELECT time, confidence AS conf_ppg_ondemand "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_ondemand' "
                "AND confidence IS NOT NULL ORDER BY time",
                "SELECT time, confidence AS conf_ppg_continuous "
                "FROM biofizic_test_ppg_only_state "
                "WHERE $__timeFilter(time) AND source = 'ppg_only_continuous' "
                "AND confidence IS NOT NULL ORDER BY time",
                "SELECT time, signal_quality AS sq_production FROM biofizic_live "
                "WHERE $__timeFilter(time) ORDER BY time",
            ],
            min_v=0, max_v=1, decimals=2,
        ),
    ]
    return dashboard_meta(
        "biofizic-cardiac-comparator",
        "Biofizic Cardiac Comparator (PPG ond/cont vs HR cont)",
        ["biofizic", "test", "comparator"],
        panels,
        version=2,
        refresh="1s",
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
        "biofizic-cardiac-comparator.json": build_cardiac_comparator_dashboard(),
    }
    for name, body in dashboards.items():
        path = OUT / name
        path.write_text(json.dumps(body, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"Wrote {path}")


if __name__ == "__main__":
    main()
