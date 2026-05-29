#!/usr/bin/env python3
"""
MQTT to InfluxDB 3 Core logger.
Subscribes to biofizic topics and writes measurements to local InfluxDB 3 Core.

Usage:
    pip install influxdb-client paho-mqtt
    python mqtt_logger.py
    python mqtt_logger.py --url http://localhost:8181 --database biofizic
"""

import sys
from pathlib import Path

_sys_root = Path(__file__).resolve().parents[1]
if str(_sys_root) not in sys.path:
    sys.path.insert(0, str(_sys_root))

import biofizic._bootstrap  # noqa: F401
import argparse
import json
import logging
import queue
import threading
import time
from datetime import datetime, timezone

import paho.mqtt.client as mqtt
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mqtt_logger")

# ── Configurare topicuri → campuri InfluxDB ────────────────────────────────────

# Numeric fields per topic (Influx measurement = topic with / replaced by _).
FLOAT_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "arousal_10", "arousal_pct", "mean_hr", "rmssd", "stress_index",
        "baseline_si", "z_si", "z_si_filtered", "kalman_gain",
        "confidence", "signal_quality", "artifact_rate", "motion_energy",
        "rmssd_w30", "rmssd_w60", "rmssd_w90", "stress_index_w30",
        "stress_index_w60", "stress_index_w90", "window_sec",
    ],
    "biofizic/state/live": [
        "arousal_10", "arousal_10_raw", "arousal_pct", "mean_hr", "rmssd",
        "stress_index", "baseline_si", "z_si", "signal_quality", "artifact_rate",
        "motion_energy", "window_sec", "ibi_buffer_size",
    ],
    "biofizic/state/windows": [
        "ibi_buffer_size",
        "w30_rmssd", "w30_sdnn", "w30_pnn50", "w30_stress_index", "w30_mean_hr",
        "w30_ibi_count", "w30_covered_seconds",
        "w60_rmssd", "w60_sdnn", "w60_pnn50", "w60_stress_index", "w60_mean_hr",
        "w60_ibi_count", "w60_covered_seconds",
        "w90_rmssd", "w90_sdnn", "w90_pnn50", "w90_stress_index", "w90_mean_hr",
        "w90_ibi_count", "w90_covered_seconds",
    ],
}

# String fields (FlightSQL queryable; tags alone may not appear in schema)
STRING_FIELDS: dict[str, list[str]] = {}

# String fields stored as InfluxDB tags (used for Grafana filtering / coloring).
TAG_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "emotion", "emotion_baseline", "motion_state", "why",
        "dominant_channel", "decision_fidelity",
    ],
    "biofizic/state/live": [
        "emotion", "motion_state", "data_quality", "window_used",
    ],
    "biofizic/state/windows": [
        "motion_state", "w30_quality", "w60_quality", "w90_quality",
    ],
}

# Boolean fields written as 0/1 floats so Grafana can plot them on axes.
BOOL_FIELDS: dict[str, list[str]] = {
    "biofizic/state": [
        "profile_ready",
        "baseline_ready",
        "labels_agree",
        "alert",
    ],
    "biofizic/state/live": [
        "live",
        "profile_ready",
        "baseline_ready",
    ],
    "biofizic/state/windows": ["baseline_ready"],
}

# Pure aligned live stream (1 Hz, ts_anchor) — Live Sync / Reliability boards.
FLOAT_FIELDS["biofizic/live"] = [
    "hr_sdk", "mean_hr", "rmssd", "stress_index",
    "z_hrv", "z_hr", "hrv_weight", "z_filtered", "kalman_gain", "arousal_10",
    "confidence", "signal_quality", "artifact_rate",
    "acc_rms", "acc_p90", "acc_std", "gyro_rms", "gyro_p90", "gyro_std", "acc_band_cardiac",
]
TAG_FIELDS["biofizic/live"] = ["motion_state", "dominant_channel", "decision_fidelity"]
BOOL_FIELDS["biofizic/live"] = ["alert", "baseline_ready"]

# Parallel research/legacy engines (never feed VR; for comparison dashboards).
FLOAT_FIELDS.update({
    "biofizic/legacy/wesad": ["p_stress"],
    "biofizic/legacy/valence": ["valence", "rmssd_z", "ppa_z"],
    "biofizic/legacy/ppg": ["n_peaks", "ppa", "ppa_z", "sample_rate_hz", "ibi_recon_mean"],
    # Respiration comparator: RSA-from-IBI vs PPG-amplitude, side by side, plus
    # the agreement when both are confident (see legacy/respiration_compare).
    "biofizic/legacy/resp": [
        "rsa_bpm", "rsa_conf", "rsa_prom",
        "ppg_bpm", "ppg_conf", "ppg_prom",
        "agree_bpm_diff",
    ],
    "biofizic/legacy/valence_fd": [
        "bf", "fhf", "shf", "bf_n", "fhf_n", "shf_n",
        "fhf_bf", "shf_bf", "shf_fhf", "f0_hz",
    ],
})

ALL_TOPICS = list(FLOAT_FIELDS.keys()) + [
    "biofizic/acquisition/batch",
    # Cardiac comparator (test_engine): raw PPG @100/25 Hz + derived HR/RMSSD
    # per source. Wildcards keep the subscribe list tight while topic dispatch
    # happens in _on_message.
    "biofizic/test/ppg_ondemand",
    "biofizic/test/ppg_continuous",
    # Covers both the lightweight comparator (ppg_ondemand / ppg_continuous /
    # hr_continuous) AND the full PPG-only PhysiologyPipeline output
    # (ppg_only_ondemand / ppg_only_continuous). Dispatch in _on_message
    # routes to the right handler based on the leaf topic name.
    "biofizic/test/derived/+",
]

# QoS 1 for low-rate epoch decisions so they survive MQTT reconnects.
TOPIC_QOS: dict[str, int] = {
    "biofizic/state": 1,
}

# Short keepalive so a dropped/stale connection to the (remote) broker is
# detected and auto-reconnected within ~1.5x this, not minutes. The logger does
# ALL InfluxDB writes, so a slow reconnect = a multi-minute data gap.
MQTT_KEEPALIVE_SEC = 20

FLOAT_FIELDS.update({
    "biofizic/acquisition/batch": [
        "hr", "skin_temp", "ambient_temp",
        "acc_rms", "acc_p90", "acc_std", "gyro_rms", "gyro_p90", "gyro_std",
        "acc_band_cardiac",
        # Atomic-sync diagnostics, plotted by biofizic-stream-sync dashboard.
        "ts_publish", "ts_anchor", "anchor_delay_ms", "skin_temp_age_ms",
        "seq", "ibi_count",
    ],
})


# Measurements that only get written when a research/legacy toggle is on. We
# seed each with one zero row at startup so the FlightSQL table EXISTS — Grafana
# then shows "No data" instead of a hard "table not found" error in those panels.
SEED_MEASUREMENTS: dict[str, list[str]] = {
    "biofizic_legacy_wesad": ["p_stress"],
    "biofizic_legacy_valence": ["valence", "rmssd_z", "ppa_z"],
    "biofizic_legacy_ppg": ["n_peaks", "ppa", "ppa_z", "sample_rate_hz", "ibi_recon_mean"],
    "biofizic_legacy_resp": [
        "rsa_bpm", "rsa_conf", "rsa_prom",
        "ppg_bpm", "ppg_conf", "ppg_prom", "agree_bpm_diff",
    ],
    "biofizic_legacy_valence_fd": [
        "bf", "fhf", "shf", "bf_n", "fhf_n", "shf_n",
        "fhf_bf", "shf_bf", "shf_fhf", "f0_hz",
    ],
    "biofizic_all_data_live": ["ppg_green", "ppg_ir", "ibi_ms", "ppg_peak"],
    # Cardiac comparator (test_engine + raw PPG sources). Seeded so Grafana
    # shows "No data" instead of "table not found" before the first publish.
    "biofizic_test_ppg_ondemand": ["green", "ir", "red"],
    "biofizic_test_ppg_continuous": ["green", "ir", "red"],
    "biofizic_test_derived": [
        "hr_bpm", "rmssd_ms", "sdnn_ms", "ibi_count",
        "peak_count", "sample_rate_hz", "artifact_rate",
    ],
    # Full PPG-only PhysiologyPipeline state (test_engine). Mirrors the keys
    # mqtt_logger writes for biofizic/state/live so Grafana can overlay them.
    "biofizic_test_ppg_only_state": [
        "arousal_10", "stress_index", "rmssd", "mean_hr",
        "z_si", "z_hr", "z_si_filtered", "hrv_weight",
        "confidence", "signal_quality", "artifact_rate",
        "kalman_gain", "ibi_count", "ibi_buffer_size",
    ],
}


def topic_to_measurement(topic: str) -> str:
    return topic.replace("/", "_")


def flatten_windows(payload: dict) -> dict:
    """Flatten nested windows JSON for Influx field/tag mapping."""
    flat: dict = {
        "baseline_ready": payload.get("baseline_ready", False),
        "ibi_buffer_size": payload.get("ibi_buffer_size", 0),
        "motion_state": payload.get("motion_state"),
    }
    for window_label, fields in payload.get("windows", {}).items():
        if not isinstance(fields, dict):
            continue
        for field_name, value in fields.items():
            key = f"{window_label}_{field_name}"
            if field_name == "quality":
                flat[key] = value
            elif value is not None:
                flat[key] = value
    return flat


def flatten_acquisition(payload: dict) -> dict:
    """
    Flatten the nested acquisition/batch payload for InfluxDB. Adds two
    derived diagnostic fields that the stream-sync dashboard plots:

      anchor_delay_ms = ts_anchor - ts_publish
          How far ahead of the publish moment the freshest stream sample is.
          A positive value confirms ts_anchor is acting as the atomic anchor.
      skin_temp_age_ms = ts_publish - skin_temp_ts
          How old the most recent skin temperature reading is at publish.
          Useful to see that slow-rate streams are still inside the batch.
    """
    ts_publish = payload.get("ts_publish")
    ts_anchor = payload.get("ts_anchor")
    skin_temp_ts = payload.get("skin_temp_ts")
    flat: dict = {
        "ts": ts_publish or ts_anchor,
        "ts_publish": ts_publish,
        "ts_anchor": ts_anchor,
        "hr": payload.get("hr"),
        "skin_temp": payload.get("skin_temp"),
        "ambient_temp": payload.get("ambient_temp"),
        "display_on": payload.get("display_on"),
        "seq": payload.get("seq"),
    }
    if isinstance(ts_publish, (int, float)) and isinstance(ts_anchor, (int, float)):
        flat["anchor_delay_ms"] = int(ts_anchor) - int(ts_publish)
    if isinstance(ts_publish, (int, float)) and isinstance(skin_temp_ts, (int, float)) and skin_temp_ts > 0:
        flat["skin_temp_age_ms"] = int(ts_publish) - int(skin_temp_ts)
    motion = payload.get("motion") or {}
    for key in (
        "acc_rms", "acc_p90", "acc_std", "gyro_rms", "gyro_p90", "gyro_std",
        "acc_band_cardiac",
    ):
        if key in motion:
            flat[key] = motion[key]
    ibi = payload.get("ibi") or {}
    if isinstance(ibi.get("ms"), list):
        flat["ibi_count"] = len(ibi["ms"])
    return flat


class MqttInfluxLogger:

    def __init__(self, broker: str, broker_port: int,
                 influx_url: str, influx_database: str):

        self.bucket    = influx_database
        self._msgs_ok  = 0
        self._msgs_err = 0
        self._msgs_recv = 0            # MQTT messages received
        self._last_msg_ts = time.time()
        self._seeded   = False
        # Decouple InfluxDB writes from the MQTT callback thread: on_message just
        # enqueues (fast, non-blocking); a dedicated writer thread drains and
        # writes in batches. Synchronous writes on the MQTT thread blocked it and
        # collapsed throughput; the old ASYNCHRONOUS worker stalled silently.
        self._wq: queue.Queue = queue.Queue(maxsize=20000)
        self._dropped = 0

        # InfluxDB 3 Core: token="ignored", org="ignored", bucket=database name.
        # SYNCHRONOUS writes: the ASYNCHRONOUS worker stalled silently against
        # InfluxDB 3 Core (and its success/error callbacks never fired), which
        # froze all logging while MQTT stayed connected. Synchronous writes are
        # inline + raise on error, so failures are visible and recoverable.
        self._influx = InfluxDBClient(
            url=influx_url, token="ignored", org="ignored"
        )
        self.write_api = self._influx.write_api(write_options=SYNCHRONOUS)
        log.info(f"InfluxDB 3 Core: {influx_url}  database={influx_database} (sync writes)")

        import uuid
        self.client = mqtt.Client(
            client_id=f"biofizic_influx_logger_{uuid.uuid4().hex[:8]}",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect    = self._on_connect
        self.client.on_message    = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.client.reconnect_delay_set(min_delay=1, max_delay=15)
        self.broker       = broker
        self.broker_port  = broker_port

    def _write(self, record) -> None:
        """Enqueue a Point or list[Point] for the writer thread (non-blocking)."""
        if not record:
            return
        try:
            self._wq.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def _writer_loop(self) -> None:
        """Drain the queue and write to InfluxDB in batches, off the MQTT thread.
        Resilient: a failed write is logged and skipped, never kills the thread."""
        while True:
            try:
                item = self._wq.get(timeout=0.5)
            except queue.Empty:
                continue
            batch: list = item if isinstance(item, list) else [item]
            while len(batch) < 1000:
                try:
                    nxt = self._wq.get_nowait()
                except queue.Empty:
                    break
                batch.extend(nxt if isinstance(nxt, list) else [nxt])
            if not batch:
                continue
            try:
                self.write_api.write(bucket=self.bucket, record=batch)
                self._msgs_ok += len(batch)
            except Exception as e:
                self._msgs_err += 1
                log.warning("InfluxDB write error (%d pts): %s", len(batch), e)

    def start(self) -> None:
        log.info(f"Conectare MQTT {self.broker}:{self.broker_port}")
        self.client.connect(self.broker, self.broker_port, keepalive=MQTT_KEEPALIVE_SEC)
        threading.Thread(target=self._writer_loop, daemon=True).start()
        threading.Thread(target=self._heartbeat, daemon=True).start()
        try:
            self.client.loop_forever()
        finally:
            self.write_api.close()
            self._influx.close()

    def _heartbeat(self, period_s: int = 30) -> None:
        """Every period_s, log message flow so a SILENT stall (connected but not
        receiving, or InfluxDB writer stuck) is visible with a timestamp — these
        do NOT trigger on_disconnect, so the heartbeat is how we catch them."""
        last_recv = 0
        last_ok = 0
        while True:
            time.sleep(period_s)
            age = time.time() - self._last_msg_ts
            d_recv = self._msgs_recv - last_recv
            d_ok = self._msgs_ok - last_ok
            last_recv, last_ok = self._msgs_recv, self._msgs_ok
            if d_recv == 0:
                log.warning(
                    "HEARTBEAT STALL: 0 MQTT msgs in %ds (last %.0fs ago) — "
                    "connected but not receiving (broker/subscription).", period_s, age,
                )
            elif d_ok == 0:
                log.warning(
                    "HEARTBEAT STALL: received %d msgs but 0 InfluxDB writes confirmed "
                    "in %ds — write worker stuck. (recv=%d ok=%d err=%d)",
                    d_recv, period_s, self._msgs_recv, self._msgs_ok, self._msgs_err,
                )
            else:
                log.info(
                    "HEARTBEAT ok: +%d msgs / +%d pts in %ds (recv=%d ok=%d err=%d "
                    "queue=%d dropped=%d)",
                    d_recv, d_ok, period_s, self._msgs_recv, self._msgs_ok,
                    self._msgs_err, self._wq.qsize(), self._dropped,
                )

    def _on_connect(self, client, userdata, flags, rc, props=None):
        if rc == 0:
            for t in ALL_TOPICS:
                client.subscribe(t, qos=TOPIC_QOS.get(t, 0))
            log.info(f"MQTT conectat — subscris la {len(ALL_TOPICS)} topicuri")
            self._seed_measurements()
        else:
            log.error(f"MQTT connect failed rc={rc}")

    def _seed_measurements(self) -> None:
        """Create the legacy/research tables (one zero row each) so Grafana shows
        'No data' instead of a FlightSQL 'table not found' error until the
        corresponding toggle is turned on."""
        if self._seeded:
            return
        self._seeded = True
        dt = datetime.now(timezone.utc)
        for measurement, fields in SEED_MEASUREMENTS.items():
            point = Point(measurement).time(dt)
            # biofizic_test_derived is tag-partitioned by source — seed one row
            # per known source so the `source` column exists in the FlightSQL
            # schema before the first real publish (otherwise Grafana panels
            # that filter WHERE source = '...' fail with "no such column").
            if measurement == "biofizic_test_derived":
                for src in ("ppg_ondemand", "ppg_continuous", "hr_continuous"):
                    seed = Point(measurement).time(dt).tag("source", src)
                    for field in fields:
                        seed.field(field, 0.0)
                    self._write(seed)
                continue
            if measurement == "biofizic_test_ppg_only_state":
                for src in ("ppg_only_ondemand", "ppg_only_continuous"):
                    seed = Point(measurement).time(dt).tag("source", src)
                    for field in fields:
                        seed.field(field, 0.0)
                    self._write(seed)
                continue
            for field in fields:
                point.field(field, 0.0)
            self._write(point)
        log.info(f"Seeded {len(SEED_MEASUREMENTS)} legacy measurements (tables exist)")

    def _on_disconnect(self, client, userdata, flags, rc, props=None):
        log.warning(f"MQTT deconectat (rc={rc})")

    @staticmethod
    def _anchor_fn(sample_ts: list[float]):
        """Return a function mapping a watch-clock sample ts -> a server-clock
        datetime, anchoring the batch's newest sample to 'now' and preserving
        intra-batch spacing. This makes the raw wave land at "now" regardless of
        watch clock skew (which can be minutes off)."""
        valid = [t for t in sample_ts if isinstance(t, (int, float)) and t > 0]
        if not valid:
            return None
        newest = max(valid)
        now_ms = datetime.now(timezone.utc).timestamp() * 1000.0

        def to_dt(t: float):
            return datetime.fromtimestamp((now_ms - (newest - t)) / 1000.0, tz=timezone.utc)

        return to_dt

    def _write_all_data_live(self, payload: dict) -> None:
        """Unroll raw PPG samples and IBI beats to per-sample points, anchored to
        server time (newest sample == now), preserving spacing."""
        ppg = payload.get("ppg") or {}
        green = ppg.get("green") or []
        ir = ppg.get("ir") or []
        ppg_ts = ppg.get("ts_ms") or []
        ibi = payload.get("ibi") or {}
        ibi_ms = ibi.get("ms") or []
        ibi_ts = ibi.get("ts") or []

        to_dt = self._anchor_fn(list(ppg_ts) + list(ibi_ts))
        if to_dt is None:
            return

        points: list[Point] = []
        for i, t in enumerate(ppg_ts):
            if not isinstance(t, (int, float)) or t <= 0:
                continue
            p = Point("biofizic_all_data_live").time(to_dt(t))
            wrote = False
            if i < len(green):
                p.field("ppg_green", float(green[i])); wrote = True
            if i < len(ir):
                p.field("ppg_ir", float(ir[i])); wrote = True
            if wrote:
                points.append(p)
        for ms, t in zip(ibi_ms, ibi_ts):
            if isinstance(t, (int, float)) and t > 0:
                points.append(
                    Point("biofizic_all_data_live").time(to_dt(t)).field("ibi_ms", float(ms))
                )
        self._write(points)  # one batched synchronous write

    def _write_ppg_peaks(self, payload: dict) -> None:
        """Mark detected PPG peaks on the ALL DATA LIVE wave (server-anchored)."""
        peaks = payload.get("peak_ts") or []
        to_dt = self._anchor_fn(list(peaks))
        if to_dt is None:
            return
        points = [
            Point("biofizic_all_data_live").time(to_dt(t)).field("ppg_peak", 1.0)
            for t in peaks
            if isinstance(t, (int, float)) and t > 0
        ]
        self._write(points)

    def _write_test_ppg_raw(self, measurement: str, payload: dict) -> None:
        """Unroll one raw PPG batch (test/ppg_ondemand or test/ppg_continuous) to
        per-sample points. Same _anchor_fn re-mapping as _write_all_data_live so
        the watch clock skew never lands these in the past."""
        samples = payload.get("samples")
        if not isinstance(samples, list) or not samples:
            return
        ts_list = [s.get("ts") for s in samples if isinstance(s, dict)]
        to_dt = self._anchor_fn(ts_list)
        if to_dt is None:
            return
        points: list[Point] = []
        for s in samples:
            if not isinstance(s, dict):
                continue
            t = s.get("ts")
            if not isinstance(t, (int, float)) or t <= 0:
                continue
            p = Point(measurement).time(to_dt(float(t)))
            wrote = False
            for src_field, dst_field in (("green", "green"), ("ir", "ir"), ("red", "red")):
                v = s.get(src_field)
                if v is None or v == "null":
                    continue
                try:
                    p.field(dst_field, float(v))
                    wrote = True
                except (TypeError, ValueError):
                    pass
            if wrote:
                points.append(p)
        self._write(points)

    def _write_test_ppg_only_state(self, payload: dict) -> None:
        """One row per PPG-only PhysiologyPipeline tick (test_engine), tagged
        with source so a single Grafana query can GROUP BY source and overlay
        ppg_only_ondemand vs ppg_only_continuous against production."""
        ts_ms = payload.get("ts")
        if isinstance(ts_ms, (int, float)) and ts_ms > 1e12:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        source = payload.get("source")
        if not source:
            return
        p = Point("biofizic_test_ppg_only_state").time(dt).tag("source", str(source))
        wrote = False
        for f in ("arousal_10", "stress_index", "rmssd", "mean_hr",
                  "z_si", "z_hr", "z_si_filtered", "hrv_weight",
                  "confidence", "signal_quality", "artifact_rate",
                  "kalman_gain", "ibi_count", "ibi_buffer_size"):
            v = payload.get(f)
            if v is None:
                continue
            try:
                p.field(f, float(v))
                wrote = True
            except (TypeError, ValueError):
                pass
        if wrote:
            self._write(p)

    def _write_test_derived(self, payload: dict) -> None:
        """One row per cardiac source (ppg_ondemand / ppg_continuous /
        hr_continuous). Tag=source so a Grafana panel can GROUP BY source and
        overlay three series in a single query."""
        ts_ms = payload.get("ts")
        if isinstance(ts_ms, (int, float)) and ts_ms > 1e12:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)
        source = payload.get("source")
        if not source:
            return
        p = Point("biofizic_test_derived").time(dt).tag("source", str(source))
        wrote = False
        for f in ("hr_bpm", "rmssd_ms", "sdnn_ms", "ibi_count", "peak_count",
                  "sample_rate_hz", "artifact_rate"):
            v = payload.get(f)
            if v is None:
                continue
            try:
                p.field(f, float(v))
                wrote = True
            except (TypeError, ValueError):
                pass
        if wrote:
            self._write(p)

    def _on_message(self, client, userdata, msg):
        self._msgs_recv += 1
        self._last_msg_ts = time.time()
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            self._msgs_err += 1
            return

        topic = msg.topic
        measurement = topic_to_measurement(topic)

        if topic == "biofizic/state/windows":
            data = flatten_windows(data)
        elif topic == "biofizic/acquisition/batch":
            # ALL DATA LIVE: unroll the raw PPG samples and IBI beats to
            # per-sample points before flattening to the summary point.
            self._write_all_data_live(data)
            data = flatten_acquisition(data)
        elif topic == "biofizic/legacy/ppg":
            # Overlay the detected PPG peaks on the raw wave dashboard.
            self._write_ppg_peaks(data)
        elif topic == "biofizic/test/ppg_ondemand":
            self._write_test_ppg_raw("biofizic_test_ppg_ondemand", data)
            return
        elif topic == "biofizic/test/ppg_continuous":
            self._write_test_ppg_raw("biofizic_test_ppg_continuous", data)
            return
        elif topic.startswith("biofizic/test/derived/ppg_only_"):
            self._write_test_ppg_only_state(data)
            return
        elif topic.startswith("biofizic/test/derived/"):
            self._write_test_derived(data)
            return

        # Timestamp din payload daca exista, altfel now
        ts_ms = data.get("ts") or data.get("ts_publish") or data.get("ts_anchor")
        if ts_ms and isinstance(ts_ms, (int, float)) and ts_ms > 1e12:
            dt = datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc)
        else:
            dt = datetime.now(timezone.utc)

        point = Point(measurement).time(dt)

        written = 0

        # Campuri numerice
        for field in FLOAT_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                try:
                    point.field(field, float(val))
                    written += 1
                except (TypeError, ValueError):
                    pass

        # Tag-uri string
        for field in TAG_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.tag(field, str(val))

        # String fields (queryable in FlightSQL)
        for field in STRING_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.field(field, str(val))
                written += 1

        # Campuri booleane → 0/1
        for field in BOOL_FIELDS.get(topic, []):
            val = data.get(field)
            if val is not None:
                point.field(field, 1.0 if val else 0.0)
                written += 1

        if written == 0:
            return

        self._write(point)


def main():
    parser = argparse.ArgumentParser(description="MQTT → InfluxDB 3 Core logger pentru Biofizic")
    parser.add_argument("--broker",   default="paxbespoke.automateflow.ro")
    parser.add_argument("--port",     type=int, default=1883)
    parser.add_argument("--url",      default="http://localhost:8181",
                        help="InfluxDB 3 Core URL (default: http://localhost:8181)")
    parser.add_argument("--database", default="biofizic",
                        help="Numele database-ului InfluxDB 3 (default: biofizic)")
    args = parser.parse_args()

    logger = MqttInfluxLogger(
        broker=args.broker,
        broker_port=args.port,
        influx_url=args.url,
        influx_database=args.database,
    )
    logger.start()


if __name__ == "__main__":
    main()
