#!/usr/bin/env python3
"""
Cardiac comparator (test/validation engine).

Two parallel observers on the same MQTT stream as production:

  Lightweight comparator (HR/RMSSD only) over three cardiac sources:
    - biofizic/test/ppg_ondemand        raw PPG @ ~100 Hz (on-demand tracker)
    - biofizic/test/ppg_continuous      raw PPG @ ~25 Hz  (continuous tracker)
    - biofizic/test/heart_rate_continuous  Samsung-processed IBI bursts
  Published on biofizic/test/derived/<source>.

  PPG-only full pipeline: two independent PhysiologyPipeline instances (one per
  raw PPG source) that run the full production decision stack (signal_quality →
  baseline → fusion → Kalman → decision_gate → arousal_mapper) on IBI derived
  from PPG peak detection instead of Samsung's processed IBI. Motion + temp are
  reused from biofizic/acquisition/batch so the only variable is the IBI source.
  Published on biofizic/test/derived/ppg_only_<source>.

Production (compute_engine) is NOT touched. This is an independent observer.
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401

import paho.mqtt.client as mqtt

from biofizic.dsp.ppg_peaks import detect_ppg_peaks
from biofizic.compute_features.hrv_metrics import compute_hrv_from_entries
from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.pipeline import PhysiologyPipeline
from biofizic.ingestion.messages import (
    AcquisitionBatchMessage,
    InterbeatIntervalEntry,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("test_engine")


# 30 s window matches the production primary HRV window so derived metrics are
# directly comparable with biofizic/state/live numbers in Grafana overlays.
WINDOW_SEC = 30
PUBLISH_INTERVAL_SEC = 1.0

# Topic mapping (watch publishes these when PUBLISH_TEST_DUMP=true).
TOPIC_PPG_OND = "biofizic/test/ppg_ondemand"
TOPIC_PPG_CONT = "biofizic/test/ppg_continuous"
TOPIC_HR_CONT = "biofizic/test/heart_rate_continuous"
# Re-used for motion + temperature in the PPG-only pipelines. Subscribing here
# is read-only; production still owns this topic.
TOPIC_ACQUISITION = "biofizic/acquisition/batch"

DERIVED_PREFIX = "biofizic/test/derived"


class InMemoryBaselineStore(RestBaselineStore):
    """RestBaselineStore that never reads or writes disk. Used by the PPG-only
    pipelines so each restart starts cold (acceptable — comparison purposes,
    fast iteration) and the production baseline file is never touched."""

    def __init__(self) -> None:
        # Skip RestBaselineStore.__init__ entirely so it doesn't try to _load
        # from disk. Reproduce the field initialisation by hand.
        from collections import deque as _deque

        from biofizic.config import BASELINE_ROBUST_WINDOW_EPOCHS

        self._path = Path("/dev/null")
        self._ln_rmssd: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_si: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_hr: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self.is_ready = False
        self.rest_observation_count = 0
        self.reported_baseline_arousal = 0.5

    def _load(self) -> None:  # pragma: no cover - never called, init bypassed
        pass

    def _save(self) -> None:
        pass


@dataclass
class _LastBatchContext:
    """The most recent biofizic/acquisition/batch, used to feed motion + temp
    + ts_anchor to the PPG-only pipelines. Without a fresh value we skip the
    PPG-only publish — there is no motion-equivalent to fabricate."""

    seen: bool = False
    ts_anchor_ms: int = 0
    acc_rms: float = 0.0
    acc_p90: float = 0.0
    acc_std: float = 0.0
    gyro_rms: float = 0.0
    gyro_p90: float = 0.0
    gyro_std: float = 0.0
    acc_band_cardiac: float = 0.0
    motion_window_ms: int = 1000
    heart_rate_bpm: float = 0.0
    display_on: bool = True
    skin_temperature_c: float = 0.0
    ambient_temperature_c: float = 0.0
    skin_temperature_ts_ms: int = 0


def _normalize_ts_ms(raw) -> int:
    """Samsung dp.timestamp is sometimes ns, sometimes ms. Mirror the watch
    normalisation in AcquisitionAssembler.normalizeSensorTimestampMs."""
    if raw is None:
        return 0
    try:
        v = int(raw)
    except (TypeError, ValueError):
        return 0
    if v <= 0:
        return 0
    if v > 1_000_000_000_000_000:
        return v // 1_000_000
    return v


def _is_status_ok(status) -> bool:
    """Match watch IbiSignalFilter.isStatusOk: null/0/-1 accepted."""
    if status is None:
        return True
    try:
        s = int(status)
    except (TypeError, ValueError):
        return True
    return s == 0 or s == -1


@dataclass
class PpgSample:
    ts_ms: int
    green: int


class _PpgBuffer:
    """Rolling ring of (ts_ms, green) for one PPG source. Drops samples older
    than WINDOW_SEC by timestamp on each append."""

    def __init__(self) -> None:
        self._q: deque[PpgSample] = deque()

    def add(self, ts_ms: int, green: int) -> None:
        self._q.append(PpgSample(ts_ms, green))
        self._trim()

    def _trim(self) -> None:
        if not self._q:
            return
        newest = self._q[-1].ts_ms
        cutoff = newest - WINDOW_SEC * 1000
        while self._q and self._q[0].ts_ms < cutoff:
            self._q.popleft()

    def snapshot(self) -> tuple[list[int], list[int]]:
        greens = [s.green for s in self._q]
        ts = [s.ts_ms for s in self._q]
        return greens, ts


class _IbiBuffer:
    """Rolling ring of InterbeatIntervalEntry for the HR continuous source.
    Each entry carries the per-beat reconstructed timestamp_ms so the artifact
    corrector can use temporal coherence."""

    def __init__(self) -> None:
        self._q: deque[InterbeatIntervalEntry] = deque()

    def extend(self, entries: list[InterbeatIntervalEntry]) -> None:
        for e in entries:
            self._q.append(e)
        self._trim()

    def _trim(self) -> None:
        if not self._q:
            return
        newest = self._q[-1].timestamp_ms or 0
        if newest <= 0:
            return
        cutoff = newest - WINDOW_SEC * 1000
        while self._q and (self._q[0].timestamp_ms or 0) < cutoff:
            self._q.popleft()

    def snapshot(self) -> list[InterbeatIntervalEntry]:
        return list(self._q)


def _walk_back_ibi_timestamps(
    intervals_ms: list[int], anchor_ts_ms: int
) -> list[InterbeatIntervalEntry]:
    """Reconstruct per-beat timestamps walking backwards from the burst anchor.
    Same algorithm as AcquisitionAssembler.buildIbiTimestamps on the watch."""
    out: list[InterbeatIntervalEntry] = [None] * len(intervals_ms)  # type: ignore
    end_ts = anchor_ts_ms
    for i in range(len(intervals_ms) - 1, -1, -1):
        out[i] = InterbeatIntervalEntry(interval_ms=intervals_ms[i], timestamp_ms=end_ts)
        end_ts -= intervals_ms[i]
    return out


class CardiacComparator:
    def __init__(self, broker: str, port: int) -> None:
        self.broker = broker
        self.port = port
        self.buf_ppg_ond = _PpgBuffer()
        self.buf_ppg_cont = _PpgBuffer()
        self.buf_hr_ibi = _IbiBuffer()
        # PPG-only full pipelines (one per raw PPG source). Each gets its own
        # in-memory baseline so they evolve independently from production
        # without touching data/rest_baseline.json. The .baseline attribute is
        # a public dataclass field on PhysiologyPipeline (pipeline.py:68), so
        # replacing it after construction is safe.
        self.ppg_ond_pipeline = PhysiologyPipeline()
        self.ppg_ond_pipeline.baseline = InMemoryBaselineStore()
        self.ppg_cont_pipeline = PhysiologyPipeline()
        self.ppg_cont_pipeline.baseline = InMemoryBaselineStore()
        self._last_batch = _LastBatchContext()
        self._ppg_only_seq = 0
        self._stop = threading.Event()

        self.client = mqtt.Client(
            client_id="biofizic_test_engine",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect

    # ── lifecycle ────────────────────────────────────────────────────────────

    def start(self) -> None:
        log.info("connecting to %s:%d", self.broker, self.port)
        self.client.connect(self.broker, self.port, keepalive=20)
        threading.Thread(target=self._tick_loop, daemon=True).start()
        self.client.loop_forever()

    def stop(self) -> None:
        self._stop.set()
        try:
            self.client.disconnect()
        except Exception:
            pass

    def _on_disconnect(self, client, userdata, flags, rc, props=None) -> None:
        if rc:
            log.warning("MQTT disconnected unexpectedly (rc=%s)", rc)

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc != 0:
            log.error("MQTT connect rc=%s", rc)
            return
        for topic in (TOPIC_PPG_OND, TOPIC_PPG_CONT, TOPIC_HR_CONT, TOPIC_ACQUISITION):
            client.subscribe(topic, qos=0)
        log.info(
            "subscribed: ppg_ondemand, ppg_continuous, heart_rate_continuous, "
            "acquisition/batch (window=%ds, publish=%.1fs)",
            WINDOW_SEC, PUBLISH_INTERVAL_SEC,
        )

    # ── message ingestion ────────────────────────────────────────────────────

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            return

        topic = msg.topic
        # acquisition/batch is the only message with a different schema (no
        # "samples" array): it's the production atomic payload v2.
        if topic == TOPIC_ACQUISITION:
            self._ingest_acquisition(data)
            return

        samples = data.get("samples")
        if not isinstance(samples, list) or not samples:
            return
        if topic == TOPIC_PPG_OND:
            self._ingest_ppg(samples, self.buf_ppg_ond)
        elif topic == TOPIC_PPG_CONT:
            self._ingest_ppg(samples, self.buf_ppg_cont)
        elif topic == TOPIC_HR_CONT:
            self._ingest_hr(samples)

    def _ingest_acquisition(self, data: dict) -> None:
        """Cache the watch acquisition payload so the PPG-only pipelines can
        reuse motion + temp + ts_anchor when they construct their synthetic
        batches. We never feed the production IBI into the PPG-only pipelines."""
        ts_pub = data.get("ts_publish") or data.get("ts") or 0
        ts_anchor = data.get("ts_anchor") or ts_pub
        try:
            ts_anchor = int(ts_anchor)
        except (TypeError, ValueError):
            return
        if ts_anchor <= 0:
            return
        motion = data.get("motion") or {}
        lb = self._last_batch
        lb.seen = True
        lb.ts_anchor_ms = ts_anchor
        lb.acc_rms = float(motion.get("acc_rms") or data.get("acc_rms") or 0)
        lb.acc_p90 = float(motion.get("acc_p90") or data.get("acc_p90") or 0)
        lb.acc_std = float(motion.get("acc_std") or data.get("acc_std") or 0)
        lb.gyro_rms = float(motion.get("gyro_rms") or data.get("gyro_rms") or 0)
        lb.gyro_p90 = float(motion.get("gyro_p90") or data.get("gyro_p90") or 0)
        lb.gyro_std = float(motion.get("gyro_std") or data.get("gyro_std") or 0)
        lb.acc_band_cardiac = float(motion.get("acc_band_cardiac") or 0)
        lb.motion_window_ms = int(motion.get("window_ms") or 1000)
        lb.heart_rate_bpm = float(data.get("hr") or 0)
        lb.display_on = bool(data.get("display_on", data.get("displayOn", True)))
        lb.skin_temperature_c = float(data.get("skin_temp") or 0)
        lb.ambient_temperature_c = float(data.get("ambient_temp") or 0)
        lb.skin_temperature_ts_ms = int(data.get("skin_temp_ts") or 0)

    @staticmethod
    def _ingest_ppg(samples: list, buf: _PpgBuffer) -> None:
        for s in samples:
            if not isinstance(s, dict):
                continue
            ts = _normalize_ts_ms(s.get("ts"))
            green = s.get("green")
            if ts <= 0 or green is None:
                continue
            try:
                buf.add(ts, int(green))
            except (TypeError, ValueError):
                continue

    def _ingest_hr(self, samples: list) -> None:
        for s in samples:
            if not isinstance(s, dict):
                continue
            anchor = _normalize_ts_ms(s.get("ts"))
            ibi_list = s.get("ibi") or []
            ibi_status = s.get("ibi_status") or []
            if anchor <= 0 or not ibi_list:
                continue
            accepted: list[int] = []
            for i, raw in enumerate(ibi_list):
                status = ibi_status[i] if i < len(ibi_status) else None
                if not _is_status_ok(status):
                    continue
                try:
                    val = int(raw)
                except (TypeError, ValueError):
                    continue
                accepted.append(val)
            if not accepted:
                continue
            entries = _walk_back_ibi_timestamps(accepted, anchor)
            self.buf_hr_ibi.extend(entries)

    # ── periodic compute + publish ───────────────────────────────────────────

    def _tick_loop(self) -> None:
        while not self._stop.is_set():
            t0 = time.time()
            try:
                self._publish_all_sources()
            except Exception:
                log.exception("tick failed")
            elapsed = time.time() - t0
            time.sleep(max(0.0, PUBLISH_INTERVAL_SEC - elapsed))

    def _publish_all_sources(self) -> None:
        ts_ms = int(time.time() * 1000)
        # Lightweight HR/RMSSD-only comparator (3 sources).
        self._publish_ppg_source("ppg_ondemand", self.buf_ppg_ond, ts_ms)
        self._publish_ppg_source("ppg_continuous", self.buf_ppg_cont, ts_ms)
        self._publish_hr_source(ts_ms)
        # Full PPG-only PhysiologyPipeline (2 sources). Needs motion context
        # from the last acquisition/batch; skip silently until one arrives.
        if self._last_batch.seen:
            self._publish_ppg_only_source(
                "ppg_only_ondemand", self.buf_ppg_ond, self.ppg_ond_pipeline, ts_ms
            )
            self._publish_ppg_only_source(
                "ppg_only_continuous", self.buf_ppg_cont, self.ppg_cont_pipeline, ts_ms
            )

    def _publish_ppg_source(self, source: str, buf: _PpgBuffer, ts_ms: int) -> None:
        greens, ts = buf.snapshot()
        if not greens:
            return
        result = detect_ppg_peaks(greens, ts)
        if result.n_peaks < 2 or len(result.reconstructed_ibi_ms) < 1:
            return
        # Each reconstructed_ibi_ms[i] ends at peak_timestamps_ms[i+1].
        peak_ts = result.peak_timestamps_ms
        entries = [
            InterbeatIntervalEntry(
                interval_ms=int(result.reconstructed_ibi_ms[i]),
                timestamp_ms=int(peak_ts[i + 1]),
            )
            for i in range(len(result.reconstructed_ibi_ms))
        ]
        hrv = compute_hrv_from_entries(entries)
        if hrv is None:
            return
        self._publish_derived(
            source=source,
            ts_ms=ts_ms,
            hr_bpm=hrv.mean_heart_rate_bpm,
            rmssd_ms=hrv.rmssd_ms,
            sdnn_ms=hrv.sdnn_ms,
            ibi_count=hrv.beat_count,
            peak_count=result.n_peaks,
            sample_rate_hz=result.sample_rate_hz,
            artifact_rate=hrv.artifact_rate,
        )

    def _publish_hr_source(self, ts_ms: int) -> None:
        entries = self.buf_hr_ibi.snapshot()
        if len(entries) < 2:
            return
        hrv = compute_hrv_from_entries(entries)
        if hrv is None:
            return
        self._publish_derived(
            source="hr_continuous",
            ts_ms=ts_ms,
            hr_bpm=hrv.mean_heart_rate_bpm,
            rmssd_ms=hrv.rmssd_ms,
            sdnn_ms=hrv.sdnn_ms,
            ibi_count=hrv.beat_count,
            peak_count=hrv.beat_count,  # no peak detection here; IBIs come from SDK
            sample_rate_hz=0.0,
            artifact_rate=hrv.artifact_rate,
        )

    def _publish_derived(
        self,
        *,
        source: str,
        ts_ms: int,
        hr_bpm: float,
        rmssd_ms: float,
        sdnn_ms: float,
        ibi_count: int,
        peak_count: int,
        sample_rate_hz: float,
        artifact_rate: float,
    ) -> None:
        payload = {
            "ts": ts_ms,
            "source": source,
            "hr_bpm": round(hr_bpm, 1),
            "rmssd_ms": round(rmssd_ms, 1),
            "sdnn_ms": round(sdnn_ms, 1),
            "ibi_count": int(ibi_count),
            "peak_count": int(peak_count),
            "sample_rate_hz": round(sample_rate_hz, 1),
            "artifact_rate": round(artifact_rate, 3),
            "window_sec": WINDOW_SEC,
        }
        self.client.publish(f"{DERIVED_PREFIX}/{source}", json.dumps(payload), qos=0)

    # ── PPG-only full pipeline ───────────────────────────────────────────────

    def _build_synthetic_batch(
        self, ibi_entries: list[InterbeatIntervalEntry]
    ) -> AcquisitionBatchMessage:
        """Construct a fake AcquisitionBatchMessage with PPG-derived IBI and
        the most recent watch motion/temp. The IBI is the only difference vs
        the production batch — that isolates the comparison."""
        self._ppg_only_seq += 1
        lb = self._last_batch
        return AcquisitionBatchMessage(
            timestamp_publish_ms=lb.ts_anchor_ms,
            timestamp_anchor_ms=lb.ts_anchor_ms,
            sequence=self._ppg_only_seq,
            heart_rate_bpm=lb.heart_rate_bpm,
            display_on=lb.display_on,
            skin_temperature_c=lb.skin_temperature_c,
            ambient_temperature_c=lb.ambient_temperature_c,
            skin_temperature_ts_ms=lb.skin_temperature_ts_ms,
            acceleration_rms=lb.acc_rms,
            acceleration_p90=lb.acc_p90,
            acceleration_std=lb.acc_std,
            gyroscope_rms=lb.gyro_rms,
            gyroscope_p90=lb.gyro_p90,
            gyroscope_std=lb.gyro_std,
            acc_band_cardiac=lb.acc_band_cardiac,
            motion_window_ms=lb.motion_window_ms,
            ibi_intervals_ms=[e.interval_ms for e in ibi_entries],
            ibi_timestamps_ms=[int(e.timestamp_ms or 0) for e in ibi_entries],
            ibi_timestamp_source="ppg_reconstructed",
        )

    def _publish_ppg_only_source(
        self,
        source: str,
        buf: _PpgBuffer,
        pipeline: PhysiologyPipeline,
        ts_ms: int,
    ) -> None:
        """Run the full PhysiologyPipeline against PPG-derived IBI and publish
        the same payload shape as biofizic/state/live so Grafana can overlay
        production vs this engine on the same panels."""
        greens, ts = buf.snapshot()
        if not greens:
            return
        result_peaks = detect_ppg_peaks(greens, ts)
        ibi_entries: list[InterbeatIntervalEntry] = []
        if result_peaks.n_peaks >= 2 and result_peaks.reconstructed_ibi_ms:
            peak_ts = result_peaks.peak_timestamps_ms
            ibi_entries = [
                InterbeatIntervalEntry(
                    interval_ms=int(result_peaks.reconstructed_ibi_ms[i]),
                    timestamp_ms=int(peak_ts[i + 1]),
                )
                for i in range(len(result_peaks.reconstructed_ibi_ms))
            ]
        # Even with no fresh beats this tick we still feed the pipeline an
        # empty IBI batch so motion/temp keep flowing; the pipeline will fall
        # back to recently buffered IBI in its 30s window.
        synth = self._build_synthetic_batch(ibi_entries)
        pipeline.ingest_acquisition(synth)
        result = pipeline.run(
            now=time.time(),
            end_timestamp_ms=self._last_batch.ts_anchor_ms,
        )
        self._publish_ppg_only_payload(source, ts_ms, result)

    def _publish_ppg_only_payload(
        self, source: str, ts_ms: int, result
    ) -> None:
        decision = result.decision
        payload: dict = {
            "ts": ts_ms,
            "source": source,
            "engine": "test_ppg_only",
            "window_sec": WINDOW_SEC,
            "ibi_buffer_size": result.ibi_buffer_size,
            "data_quality": result.best.quality,
            "motion_state": result.motion_state,
            "signal_quality": round(result.signal_quality, 3),
            "artifact_rate": round(result.artifact_rate, 3),
            "baseline_ready": result.baseline_ready,
        }
        if decision is not None:
            payload.update({
                "arousal_10": int(decision.display_arousal_10),
                "arousal_pct": round(decision.display_arousal_10 * 10.0, 1),
                "emotion": decision.display_label,
                "emotion_baseline": decision.baseline_label,
                "labels_agree": decision.labels_agree,
                "stress_index": round(decision.kubios_stress_index, 3),
                "baseline_si": round(decision.baseline_stress_index, 3),
                "z_si": round(decision.stress_index_z_score, 2),
                "z_hr": round(decision.hr_z_score, 2),
                "z_si_filtered": round(decision.stress_index_z_filtered, 2),
                "hrv_weight": round(decision.hrv_weight, 2),
                "confidence": round(decision.decision_confidence, 3),
                "dominant_channel": decision.dominant_channel,
                "kalman_gain": round(decision.kalman_gain, 3),
                "mean_hr": round(decision.mean_heart_rate_bpm, 1),
                "rmssd": round(decision.rmssd_ms, 1),
                "alert": decision.alert,
                "ibi_count": decision.multi_window.window_30_seconds.beat_count
                if decision.multi_window and decision.multi_window.window_30_seconds
                else 0,
            })
        else:
            # Keep the schema column present in InfluxDB so panels filter cleanly.
            payload.update({
                "arousal_10": None,
                "stress_index": None,
                "rmssd": None,
                "mean_hr": None,
                "z_si_filtered": None,
                "confidence": None,
            })
        self.client.publish(f"{DERIVED_PREFIX}/{source}", json.dumps(payload), qos=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()

    comparator = CardiacComparator(args.broker, args.port)

    def shutdown(_a, _b):
        comparator.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    comparator.start()


if __name__ == "__main__":
    main()
