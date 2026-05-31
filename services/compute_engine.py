#!/usr/bin/env python3
"""
Unified compute engine: MQTT ingestion -> physiology pipeline -> state output.

Publishes three MQTT topics:
  - biofizic/state         epoch decision (1 per 30 s), retained for bootstrap
  - biofizic/state/live    smoothed live arousal (1 Hz) for the watch UI
  - biofizic/state/windows w30/w60/w90 HRV side-by-side (every 5 s) for the
                           thesis window-comparison dashboard, validation only
"""

from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import affectus._bootstrap  # noqa: F401

import argparse

import paho.mqtt.client as mqtt

from affectus.config import (
    EPOCH_PUBLISH_INTERVAL_SECONDS,
    HRV_LOOKBACK_MS,
    LIVE_AROUSAL_HYSTERESIS_TICKS,
    PRIMARY_DECISION_WINDOW_SECONDS,
    SKEW_BACKLOG_WARN_SEC,
    WINDOWS_PUBLISH_INTERVAL_SECONDS,
)
from affectus.shared.arousal_mapper import arousal_scale_10_to_label
from affectus.shared.emotion import emotion_verdict, ValenceVerdictStabilizer
from affectus.engine.pipeline import PhysiologyPipeline
from affectus.legacy import LegacyEngines, toggles as legacy_toggles
from affectus.ingestion.messages import AcquisitionBatchMessage, IbiBatchMessage
from affectus.shared.hrv.results import MultiWindowResult, WindowResult
from affectus.shared.hrv.results import PhysiologyDecision

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("compute_engine")


def _parse_acquisition(data: dict) -> AcquisitionBatchMessage | None:
    schema = int(data.get("schema", 0))
    if schema < 2:
        return None
    ts_publish = int(data.get("ts_publish") or data.get("ts") or 0)
    if ts_publish <= 0:
        return None
    motion = data.get("motion") or {}
    ibi = data.get("ibi") or {}
    ibi_ms = [int(x) for x in (ibi.get("ms") or data.get("ibi_ms") or [])]
    ibi_ts = [int(x) for x in (ibi.get("ts") or data.get("ibi_ts") or [])]
    anchor_candidates = [ts_publish]
    if ibi_ts:
        anchor_candidates.append(max(ibi_ts))
    # Raw PPG is parsed only for the research/legacy engines (toggle off in
    # production keeps this out of the hot path).
    ppg_green: list[int] = []
    ppg_ir: list[int] = []
    ppg_ts: list[int] = []
    if legacy_toggles.ENABLE_RAW_PPG:
        ppg = data.get("ppg") or {}
        ppg_green = [int(x) for x in (ppg.get("green") or [])]
        ppg_ir = [int(x) for x in (ppg.get("ir") or [])]
        ppg_ts = [int(x) for x in (ppg.get("ts_ms") or [])]
        if ppg_ts:
            anchor_candidates.append(max(ppg_ts))
    skin_ts = int(data.get("skin_temp_ts") or 0)
    if skin_ts > 0:
        anchor_candidates.append(skin_ts)
    ts_anchor = int(data.get("ts_anchor") or max(anchor_candidates))
    return AcquisitionBatchMessage(
        timestamp_publish_ms=ts_publish,
        timestamp_anchor_ms=ts_anchor,
        sequence=int(data.get("seq") or 0),
        heart_rate_bpm=float(data.get("hr") or 0),
        display_on=bool(data.get("display_on", data.get("displayOn", True))),
        skin_temperature_c=float(data.get("skin_temp") or data.get("skin_temp_c") or 0),
        ambient_temperature_c=float(data.get("ambient_temp") or data.get("ambient_temp_c") or 0),
        skin_temperature_ts_ms=skin_ts,
        acceleration_rms=float(motion.get("acc_rms") or data.get("acc_rms") or 0),
        acceleration_p90=float(motion.get("acc_p90") or data.get("acc_p90") or 0),
        acceleration_std=float(motion.get("acc_std") or data.get("acc_std") or 0),
        gyroscope_rms=float(motion.get("gyro_rms") or data.get("gyro_rms") or 0),
        gyroscope_p90=float(motion.get("gyro_p90") or data.get("gyro_p90") or 0),
        gyroscope_std=float(motion.get("gyro_std") or data.get("gyro_std") or 0),
        acc_band_cardiac=float(motion.get("acc_band_cardiac") or 0),
        motion_window_ms=int(motion.get("window_ms") or 1000),
        ibi_intervals_ms=ibi_ms,
        ibi_timestamps_ms=ibi_ts,
        ibi_timestamp_source=str(ibi.get("source") or "reconstructed"),
        ppg_green=ppg_green,
        ppg_infrared=ppg_ir,
        ppg_timestamps_ms=ppg_ts,
    )


def _parse_ecg_calibration(data: dict) -> tuple[list[int], list[int], list[int], bool]:
    """Extract (ecg_mv, timestamps_ms, lead_off, final) from an ECG calibration
    chunk {recv_ms, samples:[{ts, ecg_mv, lead_off, seq}], final}. Returns empty
    lists on a malformed payload."""
    samples = data.get("samples")
    if not isinstance(samples, list):
        return [], [], [], False
    mv: list[int] = []
    ts: list[int] = []
    lead: list[int] = []
    for p in samples:
        if not isinstance(p, dict) or "ts" not in p or "ecg_mv" not in p:
            continue
        ts.append(int(p["ts"]))
        mv.append(int(p["ecg_mv"]))
        lead.append(int(p.get("lead_off", 0)))
    return mv, ts, lead, bool(data.get("final", False))


class ComputeEngineService:
    def __init__(self, broker: str, port: int) -> None:
        self.pipeline = PhysiologyPipeline()
        self.legacy = LegacyEngines()
        self._last_live_at = 0.0
        self._last_windows_at = 0.0
        self._last_epoch_at = 0.0
        self._anchor_ms = 0  # watch ts_anchor of the current batch (set per message)
        self._last_skew_log = 0.0
        # Streak-based hysteresis for the live arousal integer. We only flip
        # the displayed value when a different raw value has appeared for
        # LIVE_AROUSAL_HYSTERESIS_TICKS consecutive live ticks. This kills
        # the 1-tick alternation that median smoothing would still let through.
        self._live_displayed_a10: int | None = None
        self._live_candidate_a10: int | None = None
        self._live_candidate_streak: int = 0
        # Calibration is a process, not an instant: a recalibrate clears the
        # baseline, then it must re-collect resting epochs before it is ready
        # again. We hold "collecting" until baseline.is_ready flips back to True,
        # then publish "done" — so the watch can show a spinner meanwhile.
        self._calibrating: bool = False
        self._baseline_was_ready: bool = False
        # Capabilities announced by the watch in the handshake (biofizic/hello),
        # keyed by client_id. The pipeline gates feature-modules by the announced
        # set when present; absent (no handshake yet) falls back to inferring
        # from the batch contents, so a pre-handshake watch still works.
        self._announced_caps: dict[str, frozenset] = {}
        # Last smoothed valence + dead-band, cached from _publish_legacy so the
        # decision payload (which is built earlier in the epoch) can publish the
        # valence and the emotion verdict for the watch. One-epoch lag is fine:
        # valence is 60 s-smoothed, the epoch is 30 s.
        self._last_valence_personal_60s: float = 0.0
        self._last_valence_deadband: float = 0.225
        self._last_valence_ready: bool = False
        self._last_valence_confidence: float = 0.0
        self._last_emotion_verdict: tuple[str, int] = ("Neutru", 0)
        # Brings the valence verdict to parity with arousal: confidence gate +
        # median smoothing + verdict hysteresis (see ValenceVerdictStabilizer).
        self._valence_stabilizer = ValenceVerdictStabilizer()
        # Rolling ECG calibration buffer (raw 500 Hz samples + timestamps + lead).
        # Filled by the biofizic/ecg/calibrate handler during a finger-hold,
        # cleared on every cmd/calibrate so each calibration starts clean.
        self._ecg_cal_mv: list[int] = []
        self._ecg_cal_ts: list[int] = []
        self._ecg_cal_lead: list[int] = []
        # Last R-peak timestamp already fed to the baseline, so each beat is
        # ingested exactly once across overlapping detection windows (no dupes).
        self._ecg_last_peak_ms: int = 0
        self.client = mqtt.Client(
            client_id="biofizic_compute",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.client.on_disconnect = self._on_disconnect
        self.broker = broker
        self.port = port

    def _on_disconnect(self, client, userdata, flags, rc, props=None) -> None:
        # rc==0 is a clean local disconnect; anything else is an unexpected drop
        # (paho loop_forever will auto-reconnect and re-run _on_connect).
        if rc:
            log.warning("MQTT disconnected unexpectedly (rc=%s) - reconnecting", rc)

    def start(self) -> None:
        # Short keepalive for fast reconnect on the flaky remote broker.
        self.client.connect(self.broker, self.port, 20)
        self.client.loop_forever()

    def stop(self) -> None:
        self.client.disconnect()

    def _handle_hello(self, client, data) -> None:
        """Validate a watch capability announcement and reply with the ack.

        On a valid announcement, cache the present capabilities (keyed by
        client_id) so the pipeline gates modules by what the device declared, and
        reply status:"ok" with the active module list. On too few sensors to
        classify, reply status:"error" with the reason — the watch shows it and
        does not stream."""
        from affectus.contract.schema import parse_capabilities, validate_announcement

        if not isinstance(data, dict):
            return
        client_id = str(data.get("client_id", "watch"))
        cap_names = data.get("capabilities") or []
        profile = str(data.get("profile", "wrist_ppg"))
        ack = validate_announcement(cap_names, profile=profile)

        if ack.status == "ok":
            self._announced_caps[client_id] = parse_capabilities(cap_names)
            log.info("handshake ok: %s announced %s -> modules %s",
                     client_id, sorted(cap_names), ack.modules_active)
        else:
            self._announced_caps.pop(client_id, None)
            log.warning("handshake rejected: %s -> %s", client_id, ack.reason)

        payload = {"ts": int(time.time() * 1000), "client_id": client_id, **ack.as_dict()}
        client.publish("biofizic/hello/ack", json.dumps(payload), qos=1)

    def _handle_ecg_calibration(self, client, data, now: float) -> None:
        """Ingest one ECG calibration chunk (raw 500 Hz Lead-I): append to the
        rolling buffer, detect R-peaks over the accumulated window, build IBI, and
        feed the arousal baseline (motion-gate-bypassed). When that locks the
        baseline during the finger-hold, publish phase="done" so the watch stops
        the spinner — the same signal the epoch loop uses, just driven by ECG."""
        if not self._calibrating:
            return  # ECG only matters during an active calibration
        mv, ts, lead, final = _parse_ecg_calibration(data)
        if mv:
            self._ecg_cal_mv.extend(mv)
            self._ecg_cal_ts.extend(ts)
            self._ecg_cal_lead.extend(lead)
            # Cap the buffer to the HRV lookback window so it can't grow unbounded
            # over a long hold (keep the most recent samples).
            max_samples = int(HRV_LOOKBACK_MS / 1000.0 * 520)  # ~ lookback @ >500 Hz
            if len(self._ecg_cal_ts) > max_samples:
                self._ecg_cal_mv = self._ecg_cal_mv[-max_samples:]
                self._ecg_cal_ts = self._ecg_cal_ts[-max_samples:]
                self._ecg_cal_lead = self._ecg_cal_lead[-max_samples:]

        from affectus.shared.dsp.ecg_peaks import detect_ecg_rpeaks

        res = detect_ecg_rpeaks(self._ecg_cal_mv, self._ecg_cal_ts, self._ecg_cal_lead)
        # Each interval is keyed by its END peak timestamp; feed only beats newer
        # than the last ingested so re-detecting over the growing buffer never
        # double-counts (the old bug: same beats ingested every chunk -> the lock
        # fired on accumulation artifacts, not real elapsed time).
        new_ts: list[int] = []
        new_ibi: list[int] = []
        for ts, ms in zip(res.peak_timestamps_ms[1:], res.reconstructed_ibi_ms):
            if ts > self._ecg_last_peak_ms:
                new_ts.append(ts)
                new_ibi.append(ms)
        if new_ibi:
            self._ecg_last_peak_ms = new_ts[-1]
            batch = IbiBatchMessage(
                timestamp_ms=new_ts[-1],
                intervals_ms=new_ibi,
                timestamps_ms=new_ts,
            )
            ready = self.pipeline.ingest_ecg_calibration(batch, now=now)
            # Re-lock signal, identical to the epoch loop's: announce "done" the
            # moment the baseline locks from ECG IBI.
            if self._calibrating and ready and not self._baseline_was_ready:
                self._calibrating = False
                self.pipeline.ecg_calibration_until = 0.0  # ECG won; PPG can resume
                self._publish_calibration(client, "Profil calibrat (ECG)", phase="done")
            self._baseline_was_ready = ready

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc != 0:
            return
        topics = [
            ("biofizic/acquisition/batch", 0),
            ("biofizic/cmd/calibrate", 1),
            # Handshake: the watch announces its sensors here; we validate and
            # reply on biofizic/hello/ack. QoS 1 so the announce/ack survive a
            # reconnect.
            ("biofizic/hello", 1),
            # ECG calibration stream (raw 500 Hz Lead-I) — only flows during a
            # finger-hold calibration; the server detects R-peaks -> IBI ->
            # baseline. Enhancement over PPG; PPG resting stays as fallback.
            ("biofizic/ecg/calibrate", 0),
        ]
        # Subscribe to the 100 Hz on-demand PPG when ANY valence engine that uses
        # the high-rate waveform is active, so each gets the sharp stream instead
        # of falling back to the 25 Hz batch (which collapses fine-HRV models).
        if any(getattr(self.legacy, name, None) is not None for name in
               ("_valence_fd", "_valence_wesad", "_valence_eevr", "_valence_case")):
            topics.append(("biofizic/ppg/ondemand", 0))
        for topic, qos in topics:
            client.subscribe(topic, qos=qos)
        log.info("Compute engine active (acquisition/batch v2)")

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            return

        topic = msg.topic
        now = time.time()

        if topic == "biofizic/ppg/ondemand":
            # 100 Hz on-demand PPG, wrapped as {"recv_ms":.., "samples":[{ts,
            # green,ir,red},..]}. Feed the valence-FD high-rate buffer; never
            # touches the decision.
            vfd = getattr(self.legacy, "_valence_fd", None)
            vw = getattr(self.legacy, "_valence_wesad", None)
            ve = getattr(self.legacy, "_valence_eevr", None)
            vc = getattr(self.legacy, "_valence_case", None)
            # Every valence engine that consumes the 100 Hz waveform must get it,
            # or it falls back to the 25 Hz batch buffer and its morphology/HRV
            # features degrade (CASE, which leans on fine HRV LF/HF, collapses to
            # a constant at 25 Hz). Feed all of them.
            consumers = [c for c in (vfd, vw, ve, vc) if c is not None]
            sample_list = data.get("samples") if isinstance(data, dict) else None
            if consumers and isinstance(sample_list, list):
                samples = [
                    (int(p["ts"]), int(p["green"]))
                    for p in sample_list
                    if isinstance(p, dict) and "ts" in p and "green" in p
                ]
                if samples:
                    for c in consumers:
                        c.ingest_ondemand_ppg(samples)
            return

        if topic == "biofizic/ecg/calibrate":
            self._handle_ecg_calibration(client, data, now)
            return

        if topic == "biofizic/hello":
            # Capability handshake: the watch announces which sensors it carries
            # ({client_id, schema, capabilities:[...], profile?}). We validate
            # against the server-defined contract, cache the result so the
            # pipeline can gate modules by the ANNOUNCED capabilities, and reply
            # on biofizic/hello/ack. The watch waits for status:"ok" before it
            # streams. NEVER feeds the decision directly.
            self._handle_hello(client, data)
            return

        if topic == "biofizic/cmd/calibrate":
            # Optional self-reports: arousal (0..1) anchors the arousal scale;
            # valence (-1..+1) feeds the polarity sign-guard; reactivity
            # (low|normal|high) is the one-time profile that scales the valence
            # dead-band. All optional — a bare calibrate still works.
            reported = data.get("reported_arousal")
            reported = float(reported) if isinstance(reported, (int, float)) else None
            reported_valence = data.get("reported_valence")
            reported_valence = (
                float(reported_valence)
                if isinstance(reported_valence, (int, float)) else None
            )
            reactivity = data.get("reactivity")
            reactivity = reactivity if reactivity in ("low", "normal", "high") else None
            self.pipeline.reset_baseline(reported, reported_valence, reactivity)
            # ECG calibration is disabled (Samsung ECG R-peak detection was
            # unreliable on the real watch). Calibration always uses the PPG
            # resting path. The ECG code stays on disk for future research.
            self.pipeline.ecg_calibration_until = 0.0
            # Recalibration re-anchors valence -> drop the verdict history too so
            # it doesn't carry the old subject's hysteresis/median into the new one.
            self._valence_stabilizer.reset()
            self._last_emotion_verdict = ("Neutru", 0)
            # Fresh ECG calibration buffer for this run (no stale finger-hold data).
            self._ecg_cal_mv = []
            self._ecg_cal_ts = []
            self._ecg_cal_lead = []
            self._ecg_last_peak_ms = 0
            self.pipeline.reset_ecg_calibration()
            # Reset clears the baseline -> not ready. Enter "collecting" and let
            # the epoch loop publish "done" once it re-locks.
            self._calibrating = True
            self._baseline_was_ready = False
            self._publish_calibration(
                client,
                "Recalibrare… stai liniștit 1–2 min",
                phase="collecting",
            )
            return

        if topic == "biofizic/acquisition/batch":
            batch = _parse_acquisition(data)
            if batch is None:
                return
            # Diagnostic: watch->server skew measured AT receipt (clock offset +
            # delivery latency). ~0 => watch clock fine and stream live; a large
            # value => watch clock behind OR a delivery backlog.
            skew_s = (now * 1000 - batch.timestamp_publish_ms) / 1000.0
            # A steady ~1-2 s is normal (batch buffering + MQTT), not a problem.
            # Only a large/growing skew means a real delivery backlog, so warn on
            # that; keep the routine value on DEBUG to stop spamming INFO.
            if skew_s > SKEW_BACKLOG_WARN_SEC:
                if now - self._last_skew_log >= 10.0:
                    self._last_skew_log = now
                    log.warning(
                        "watch->server skew = %.1f s — delivery backlog (data arriving late)",
                        skew_s,
                    )
            elif now - self._last_skew_log >= 10.0:
                self._last_skew_log = now
                log.debug("watch->server skew = %.1f s (ts_publish vs server-now)", skew_s)
            self.pipeline.ingest_acquisition(batch)
            result = self._run_and_publish(
                client,
                now=now,
                batch=batch,
                end_timestamp_ms=batch.timestamp_anchor_ms,
            )
            if self.legacy.active:
                self._publish_legacy(client, batch, result)
            return

        return

    def _run_and_publish(
        self,
        client,
        *,
        now: float,
        batch,
        end_timestamp_ms: int | None = None,
    ):
        # Stamp every published stream for this batch with the SAME server-receive
        # timestamp. All per-batch streams (HR/RMSSD/motion/z/arousal) thus share
        # one timestamp -> mutually aligned, AND land at "now" -> immune to watch
        # clock skew (the watch clock can be minutes off; we must not store at
        # watch time or the live dashboards go blank).
        self._anchor_ms = int(now * 1000)

        publish_epoch = now - self._last_epoch_at >= EPOCH_PUBLISH_INTERVAL_SECONDS
        if publish_epoch:
            self._last_epoch_at = now
        # Let _publish_legacy advance the epoch-paced verdict stabiliser only on
        # epoch ticks (every 30 s), not on every 1 Hz batch — its median/hysteresis
        # windows are sized in epochs, not seconds.
        self._is_epoch_tick = publish_epoch

        result = self.pipeline.run(
            now=now,
            end_timestamp_ms=end_timestamp_ms,
            publish_epoch=publish_epoch,
        )

        if now - self._last_live_at >= 1.0:
            self._last_live_at = now
            self._publish_live(client, result)
            self._publish_live_metrics(client, result, batch)

        if now - self._last_windows_at >= WINDOWS_PUBLISH_INTERVAL_SECONDS:
            self._last_windows_at = now
            self._publish_windows(client, result)

        if publish_epoch and result.decision:
            self._publish_state(client, result.decision, result=result)

        # Calibration progress: announce "done" the moment the baseline re-locks
        # after a recalibrate, so the watch stops the spinner.
        if self._calibrating and result.baseline_ready and not self._baseline_was_ready:
            self._calibrating = False
            self._publish_calibration(
                client, "Profil calibrat", phase="done"
            )
        self._baseline_was_ready = result.baseline_ready
        return result

    def _publish_live_metrics(self, client, result: MultiWindowResult, batch) -> None:
        """Pure aligned live stream on biofizic/live (1 Hz, ts_anchor): every raw
        and computed scalar on one clock, for the Live Sync / Reliability boards."""
        d = result.decision
        payload = {
            "ts": self._anchor_ms,
            "hr_sdk": round(batch.heart_rate_bpm, 1),
            "acc_rms": round(batch.acceleration_rms, 3),
            "acc_p90": round(batch.acceleration_p90, 3),
            "acc_std": round(batch.acceleration_std, 3),
            "gyro_rms": round(batch.gyroscope_rms, 4),
            "gyro_p90": round(batch.gyroscope_p90, 4),
            "gyro_std": round(batch.gyroscope_std, 4),
            "acc_band_cardiac": round(batch.acc_band_cardiac, 4),
            "motion_state": result.motion_state,
            "signal_quality": round(result.signal_quality, 3),
            "artifact_rate": round(result.artifact_rate, 3),
            "baseline_ready": result.baseline_ready,
        }
        if d is not None:
            payload.update({
                "mean_hr": round(d.mean_heart_rate_bpm, 1),
                "rmssd": round(d.rmssd_ms, 1),
                "stress_index": round(d.kubios_stress_index, 3),
                "z_hrv": round(d.stress_index_z_score, 2),
                "z_hr": round(d.hr_z_score, 2),
                "hrv_weight": round(d.hrv_weight, 2),
                "confidence": round(d.decision_confidence, 3),
                "dominant_channel": d.dominant_channel,
                "decision_fidelity": d.decision_fidelity,
                "z_filtered": round(d.stress_index_z_filtered, 2),
                "kalman_gain": round(d.kalman_gain, 3),
                "arousal_10": d.display_arousal_10,
                "alert": d.alert,
            })
        client.publish("biofizic/live", json.dumps(payload), qos=0)

    def _enrich_valence(self, out_dict, vbase, vsmoother, result) -> float | None:
        """Recenter one valence model's raw valence_z on the subject's personal
        resting baseline and add the calibrated fields to its published dict, the
        same way WESAD does. Used for WESAD, EEVR and CASE so all three show
        subject-relative emotion (neutral measured, not assumed 0). Returns the
        60 s-smoothed personal valence (or None if no valence_z this epoch)."""
        vz = out_dict.get("valence_z")
        if vz is None:
            return None
        is_resting = (
            result is not None
            and getattr(result, "motion_state", "") == "still"
            and getattr(result, "signal_quality", 0.0) >= 0.5
        )
        if is_resting:
            vbase.observe_resting(float(vz), now=time.time())
        personal = vbase.valence_personal(float(vz))
        smoothed = vsmoother.update(personal, time.time())
        deadband = vbase.deadband(
            reactivity_scale=self.pipeline.reactivity.deadband_scale
        )
        out_dict["valence_personal"] = round(personal, 4)
        out_dict["valence_personal_60s"] = round(smoothed, 4)
        out_dict["valence_deadband"] = round(deadband, 4)
        out_dict["neutral_valence_z"] = round(vbase.neutral_valence_z or 0.0, 4)
        out_dict["valence_baseline_ready"] = vbase.is_ready
        return smoothed

    def _publish_legacy(self, client, batch, result) -> None:
        """Publish the parallel research engines on biofizic/legacy/* (never VR)."""
        out = self.legacy.run(batch=batch, result=result, baseline=self.pipeline.baseline)
        ts = self._anchor_ms
        log.debug("legacy out: ppg=%s wesad=%s valence=%s resp=%s",
                  out.ppg is not None, out.wesad is not None,
                  out.valence is not None, out.respiration)
        if out.ppg is not None:
            client.publish("biofizic/legacy/ppg", json.dumps({"ts": ts, **out.ppg}), qos=0)
        if out.wesad is not None:
            client.publish("biofizic/legacy/wesad", json.dumps({"ts": ts, **out.wesad}), qos=0)
        if out.valence is not None:
            client.publish("biofizic/legacy/valence", json.dumps({"ts": ts, **out.valence}), qos=0)
        if out.respiration is not None:
            client.publish("biofizic/legacy/resp", json.dumps({"ts": ts, **out.respiration}), qos=0)
        if out.valence_fd is not None:
            client.publish("biofizic/legacy/valence_fd", json.dumps({"ts": ts, **out.valence_fd}), qos=0)
        if out.valence_wesad is not None:
            # Personal valence calibration: recenter the cross-device WESAD model
            # on this subject (neutral measured on resting epochs, not assumed 0).
            vbase = self.pipeline.valence_baseline
            smoothed = self._enrich_valence(
                out.valence_wesad, vbase, self.pipeline.valence_smoother, result)
            if smoothed is not None:
                # WESAD is the channel that drives the watch verdict, so it also
                # caches for the decision payload and advances the stabiliser.
                self._last_valence_personal_60s = smoothed
                self._last_valence_deadband = out.valence_wesad["valence_deadband"]
                self._last_valence_ready = vbase.is_ready
                self._last_valence_confidence = float(
                    out.valence_wesad.get("confidence", 0.0)
                )
                # Advance the verdict stabiliser ONCE per 30 s epoch (median +
                # hysteresis windows are in epochs); cached verdict persists between.
                if getattr(self, "_is_epoch_tick", False):
                    arousal_z = 0.0
                    if result is not None and result.decision is not None:
                        arousal_z = result.decision.stress_index_z_filtered
                    self._last_emotion_verdict = self._valence_stabilizer.update(
                        arousal_z, smoothed, self._last_valence_confidence,
                        out.valence_wesad["valence_deadband"], vbase.is_ready,
                    )
            client.publish("biofizic/legacy/valence_wesad", json.dumps({"ts": ts, **out.valence_wesad}), qos=0)

        # EEVR / CASE valence: observed-only comparison models, each personally
        # recentered on its OWN baseline (same as WESAD) so the dashboard shows
        # calibrated emotion for all three — not raw valence. They do NOT feed the
        # watch verdict (no stabiliser / decision cache).
        if out.valence_eevr is not None:
            self._enrich_valence(out.valence_eevr, self.pipeline.valence_baseline_eevr,
                                 self.pipeline.valence_smoother_eevr, result)
            client.publish("biofizic/legacy/valence_eevr", json.dumps({"ts": ts, **out.valence_eevr}), qos=0)
        if out.valence_case is not None:
            self._enrich_valence(out.valence_case, self.pipeline.valence_baseline_case,
                                 self.pipeline.valence_smoother_case, result)
            client.publish("biofizic/legacy/valence_case", json.dumps({"ts": ts, **out.valence_case}), qos=0)

    @staticmethod
    def _window_payload(window: WindowResult) -> dict:
        if window.quality == "unavailable":
            return {
                "rmssd": None,
                "sdnn": None,
                "pnn50": None,
                "stress_index": None,
                "mean_hr": None,
                "quality": window.quality,
                "ibi_count": window.ibi_count,
                "covered_seconds": round(window.covered_seconds, 1),
            }
        return {
            "rmssd": round(window.rmssd_ms, 1),
            "sdnn": round(window.sdnn_ms, 1),
            "pnn50": round(window.pnn50_pct, 1),
            "stress_index": round(window.kubios_stress_index, 3),
            "mean_hr": round(window.mean_hr_bpm, 1),
            "quality": window.quality,
            "ibi_count": window.ibi_count,
            "covered_seconds": round(window.covered_seconds, 1),
        }

    def _decision_payload(
        self,
        decision: PhysiologyDecision,
        *,
        live: bool,
        result: MultiWindowResult | None = None,
    ) -> dict:
        multi = decision.multi_window
        sensor = self.pipeline.state.last_sensor
        window_sec = PRIMARY_DECISION_WINDOW_SECONDS

        payload = {
            "ts": self._anchor_ms,
            "live": live,
            "engine": "compute",
            "emotion": decision.display_label,
            "emotion_baseline": decision.baseline_label,
            "labels_agree": decision.labels_agree,
            "arousal_10": decision.display_arousal_10,
            "arousal_pct": round(decision.display_arousal_10 * 10.0, 1),
            # NOTE: valence + 2D emotion verdict are NOT sent to the watch — they
            # were too noisy/low-confidence to be useful. Arousal is the only live
            # classifier on the watch. The valence research path still runs on
            # biofizic/legacy/valence_wesad (Grafana only) for offline study.
            # Multi-channel verdict confidence (HR carries it in motion), not the
            # HRV-only quality which collapses when moving. dominant_channel says
            # which signal drives the verdict so the UI can show "via HR".
            "confidence": round(decision.decision_confidence, 3),
            "dominant_channel": decision.dominant_channel,
            # "preliminary" = arousal from Kubios population zones (no personal
            # baseline yet); "calibrated" = arousal from personal-z CDF. UI can
            # show a badge so a calibrated verdict isn't confused with the cold
            # start fallback.
            "decision_fidelity": decision.decision_fidelity,
            "stress_index": round(decision.kubios_stress_index, 3),
            "baseline_si": round(decision.baseline_stress_index, 3),
            "z_si": round(decision.stress_index_z_score, 2),
            "z_si_filtered": round(decision.stress_index_z_filtered, 2),
            "kalman_gain": round(decision.kalman_gain, 3),
            "rmssd": round(decision.rmssd_ms, 1),
            "mean_hr": round(decision.mean_heart_rate_bpm, 1),
            # Signal-quality gate outputs (replaced HAR activity class + caps).
            "motion_state": decision.motion_state,
            "signal_quality": round(decision.signal_quality, 3),
            "artifact_rate": round(decision.artifact_rate, 3),
            "motion_energy": round(decision.motion_energy, 3),
            "alert": decision.alert,
            "motion_acc_rms": round(sensor.acceleration_rms, 3) if sensor else 0.0,
            "acc_window_p90": round(sensor.acceleration_p90, 3) if sensor else 0.0,
            "profile_ready": decision.baseline_ready,
            "baseline_ready": decision.baseline_ready,
            "why": decision.decision_reason,
            "window_sec": window_sec,
        }
        if result is not None:
            # Report the window that ACTUALLY drove the verdict (w60 normally,
            # w30 only during the first 60 s before w60 is computable), not a
            # hardcoded label — the windows dashboard must not misattribute the
            # decision to w30.
            payload["window_used"] = result.best_window_label
            payload["data_quality"] = result.best.quality
            payload["ibi_buffer_size"] = result.ibi_buffer_size
        if multi:
            for suffix, metrics in (
                ("w30", multi.window_30_seconds),
                ("w60", multi.window_60_seconds),
                ("w90", multi.window_90_seconds),
            ):
                if metrics and metrics.is_valid:
                    payload[f"rmssd_{suffix}"] = round(metrics.rmssd_ms, 1)
                    payload[f"stress_index_{suffix}"] = round(metrics.kubios_stress_index, 3)
        return payload

    def _publish_live(self, client, result: MultiWindowResult) -> None:
        decision = result.decision
        if decision:
            payload = self._decision_payload(decision, live=True, result=result)
            self._apply_live_arousal_smoothing(payload, decision)
        else:
            # No decision yet means we have no fresh integer arousal to publish.
            # Reset hysteresis state so a later run does not flip from a stale
            # value carried across the gap.
            self._live_displayed_a10 = None
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            payload = {
                "ts": self._anchor_ms,
                "live": True,
                "engine": "compute",
                "window_used": result.best_window_label,
                "data_quality": result.best.quality,
                "ibi_buffer_size": result.ibi_buffer_size,
                "motion_state": result.motion_state,
                "signal_quality": round(result.signal_quality, 3),
                "artifact_rate": round(result.artifact_rate, 3),
                "baseline_ready": result.baseline_ready,
                "arousal_10": None,
                "stress_index": None,
                "rmssd": None,
                "mean_hr": None,
                "window_sec": PRIMARY_DECISION_WINDOW_SECONDS,
            }
        client.publish("biofizic/state/live", json.dumps(payload), qos=0)

    def _apply_live_arousal_smoothing(
        self, payload: dict, decision: PhysiologyDecision
    ) -> None:
        """
        Apply streak-based hysteresis to the integer arousal_10 reported on
        biofizic/state/live.

        The pipeline recomputes Kubios SI every second on a rolling 30 s buffer.
        A single new IBI can shift SI just enough to cross a zone boundary,
        which flips the rounded arousal_10 by one. Without hysteresis the watch
        UI alternates 2-3-2-3 every second even when the underlying signal is
        stable. We only adopt a new displayed integer after it has been seen
        for LIVE_AROUSAL_HYSTERESIS_TICKS consecutive ticks; otherwise we keep
        the previous one. The label is recomputed from the displayed integer so
        the text on the watch stays consistent with the number.
        """
        raw_a10 = int(decision.display_arousal_10)
        displayed = self._update_live_arousal_hysteresis(raw_a10)
        payload["arousal_10"] = displayed
        payload["arousal_pct"] = round(displayed * 10.0, 1)
        payload["emotion"] = arousal_scale_10_to_label(displayed)
        # Keep the unsmoothed value visible for Grafana and debugging so we
        # can see when the hysteresis is hiding a real transition.
        payload["arousal_10_raw"] = raw_a10

    def _update_live_arousal_hysteresis(self, raw_a10: int) -> int:
        if self._live_displayed_a10 is None:
            self._live_displayed_a10 = raw_a10
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            return raw_a10
        if raw_a10 == self._live_displayed_a10:
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
            return self._live_displayed_a10
        if raw_a10 == self._live_candidate_a10:
            self._live_candidate_streak += 1
        else:
            self._live_candidate_a10 = raw_a10
            self._live_candidate_streak = 1
        if self._live_candidate_streak >= LIVE_AROUSAL_HYSTERESIS_TICKS:
            self._live_displayed_a10 = raw_a10
            self._live_candidate_a10 = None
            self._live_candidate_streak = 0
        return self._live_displayed_a10

    def _publish_windows(self, client, result: MultiWindowResult) -> None:
        payload = {
            "ts": self._anchor_ms,
            "windows": {
                "w30": self._window_payload(result.w30),
                "w60": self._window_payload(result.w60),
                "w90": self._window_payload(result.w90),
            },
            "motion_state": result.motion_state,
            "baseline_ready": result.baseline_ready,
            "ibi_buffer_size": result.ibi_buffer_size,
        }
        client.publish("biofizic/state/windows", json.dumps(payload), qos=0)

    def _publish_state(
        self,
        client,
        decision: PhysiologyDecision,
        *,
        result: MultiWindowResult | None = None,
    ) -> None:
        """
        Publish the epoch decision (every 30 s) on biofizic/state. The message
        is retained so a watch reconnecting between epochs can bootstrap from
        the last known decision without waiting up to 30 s for the next one.
        """
        payload = self._decision_payload(decision, live=False, result=result)
        client.publish("biofizic/state", json.dumps(payload), qos=1, retain=True)

    def _publish_calibration(self, client, message: str, *, phase: str = "done") -> None:
        payload = {
            "ts": int(time.time() * 1000),
            "action": "profile",
            "phase": phase,
            "message": message,
            "profile_ready": self.pipeline.baseline.is_ready,
        }
        client.publish("biofizic/calibration/status", json.dumps(payload), qos=1, retain=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--broker", default="localhost")
    parser.add_argument("--port", type=int, default=1883)
    args = parser.parse_args()
    service = ComputeEngineService(args.broker, args.port)

    def shutdown(_a, _b):
        service.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    service.start()


if __name__ == "__main__":
    main()
