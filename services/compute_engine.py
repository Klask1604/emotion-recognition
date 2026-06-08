#!/usr/bin/env python3
"""
Unified compute engine: MQTT ingestion -> physiology pipeline -> state output.

Publishes the verdict on biofizic/out/* topics:
  - biofizic/out/arousal   epoch decision (1 per 30 s), retained for bootstrap
  - biofizic/out/live      smoothed live arousal (1 Hz) for the watch UI
  - biofizic/out/windows   w30/w60/w90 HRV side-by-side (every 5 s) for the
                           thesis window-comparison dashboard, validation only
  - biofizic/out/emotion   emotion verdict: arousal-level state (Relaxat/Normal/
                           "Stres sau Entuziasm") + arousal/valence scores. The
                           polarity (Stresat vs Entuziasmat) is named only at high
                           arousal with reliable morphology; otherwise the verdict
                           reports the ambiguous zone honestly, never a guess.
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
    LIVE_AROUSAL_HYSTERESIS_TICKS,
    PRIMARY_DECISION_WINDOW_SECONDS,
    SKEW_BACKLOG_WARN_SEC,
    VALENCE_INTENSITY_GATE_Z,
    WINDOWS_PUBLISH_INTERVAL_SECONDS,
)
from affectus.topics import In, Out
from affectus.dsp.arousal_mapper import arousal_scale_10_to_label
from affectus.dsp.emotion import ValenceVerdictStabilizer
from affectus.engine.pipeline import PhysiologyPipeline
from affectus.research import ResearchEngines, toggles as research_toggles
from affectus.io.messages import AcquisitionBatchMessage
from affectus.dsp.hrv.results import MultiWindowResult, WindowResult
from affectus.dsp.hrv.results import PhysiologyDecision

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
    if research_toggles.ENABLE_RAW_PPG:
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


class ComputeEngineService:
    def __init__(self, broker: str, port: int) -> None:
        self.pipeline = PhysiologyPipeline()
        self.research = ResearchEngines()
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
        # then publish "done", so the watch can show a spinner meanwhile.
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
        # Latest VR scene context from Unity (raw cues + derived valence prior +
        # wall-clock ts). The emotion verdict uses it when fresh (<10 s) so valence
        # comes from the SCENE (honest), not the pulse. None until Unity sends one.
        self._last_context: dict | None = None
        # Brings the valence verdict to parity with arousal: confidence gate +
        # median smoothing + verdict hysteresis (see ValenceVerdictStabilizer).
        # One stabiliser PER model so each emotion verdict (WESAD/EEVR/CASE) is
        # smoothed independently and the dashboard reads a ready quadrant instead
        # of recomputing the logic in SQL (single source of truth in Python).
        # The watch's 2D verdict stabiliser (WESAD valence + arousal). The per-model
        # dashboard stabilisers (wesad/eevr/case) now live inside each ValenceTrack
        # in ResearchEngines, so there is no parallel dict to keep in sync here.
        self._valence_stabilizer = ValenceVerdictStabilizer()
        # 3-state classifier (CALM/DISCONFORT/PLACUT): the activation axis of the
        # quadrant verdict. Trained on EMOGNITION Galaxy (9 HRV+motion features +
        # personal deviation); accumulates its own rest baseline live. Published on
        # biofizic/out/emotion for the watch and Grafana.
        try:
            from affectus.research.stari.state_classifier import StateClassifier
            self._state_clf = StateClassifier()
        except Exception:
            self._state_clf = None
        # Morphology classifier (axis 2 = valence from PPG wave shape). Consumes the
        # 100 Hz PPG, downsamples to 64 Hz, separates DISCONFORT vs PLACUT (86%) where
        # HRV alone drops to 52% here. Refines valence in the activated half.
        try:
            from affectus.research.stari.morpho_classifier import MorphoClassifier
            self._morpho_clf = MorphoClassifier()
        except Exception:
            self._morpho_clf = None
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
        classify, reply status:"error" with the reason, the watch shows it and
        does not stream."""
        from affectus.contract.handshake import parse_capabilities, validate_announcement

        if not isinstance(data, dict):
            return
        client_id = str(data.get("client_id", "watch"))
        cap_names = data.get("capabilities") or []
        profile = str(data.get("profile", "wrist_ppg"))
        ack = validate_announcement(cap_names, profile=profile)

        if ack.status == "ok":
            caps = parse_capabilities(cap_names)
            self._announced_caps[client_id] = caps
            # The declared sensors are the single source of truth for which fusion
            # channels run, push them into the pipeline so decide() gates on the
            # announcement, not on a hardcoded default.
            self.pipeline.set_declared_capabilities(caps)
            log.info("handshake ok: %s announced %s -> modules %s",
                     client_id, sorted(cap_names), ack.modules_active)
        else:
            self._announced_caps.pop(client_id, None)
            log.warning("handshake rejected: %s -> %s", client_id, ack.reason)

        payload = {"ts": int(time.time() * 1000), "client_id": client_id, **ack.as_dict()}
        client.publish(Out.HELLO_ACK, json.dumps(payload), qos=1)

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc != 0:
            return
        topics = [
            (In.ACQUISITION, 0),
            (In.CMD_CALIBRATE, 1),
            # Handshake: the watch announces its sensors here; we validate and
            # reply on Out.HELLO_ACK. QoS 1 so the announce/ack survive a reconnect.
            (In.HELLO, 1),
            # User emotion feedback (a Russell quadrant tap). QoS 1 so a label is
            # never silently dropped, each one is a hand-collected training pair.
            (In.CMD_FEEDBACK, 1),
            # VR scene context from Unity (raw visual cues -> valence prior). QoS 0
            # at ~1 Hz; the verdict uses the latest if it is fresh (<10 s).
            (In.CONTEXT, 0),
        ]
        # PPG 100 Hz nu mai e un topic separat: vine in payload-ul acquisition
        # batch (a single optical tracker). The high-resolution consumers
        # (morfologie + valence FD) sunt hraniti din handler-ul batch.
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

        if topic == In.HELLO:
            # Capability handshake: the watch announces which sensors it carries
            # ({client_id, schema, capabilities:[...], profile?}). We validate
            # against the server-defined contract, cache the result so the
            # pipeline can gate modules by the ANNOUNCED capabilities, and reply
            # on biofizic/hello/ack. The watch waits for status:"ok" before it
            # streams. NEVER feeds the decision directly.
            self._handle_hello(client, data)
            return

        if topic == In.CMD_CALIBRATE:
            # Optional self-reports: arousal (0..1) anchors the arousal scale;
            # valence (-1..+1) feeds the polarity sign-guard; reactivity
            # (low|normal|high) is the one-time profile that scales the valence
            # dead-band. All optional, a bare calibrate still works.
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
            # Valence calibration lives in research now; re-anchor it on the same
            # recalibration so every model's personal neutral is re-measured.
            self.research.reset_valence_tracks(reported_valence)
            # Recalibration re-anchors valence -> drop the verdict history too so
            # it doesn't carry the old subject's hysteresis/median into the new one.
            self._valence_stabilizer.reset()
            self._last_emotion_verdict = ("Neutru", 0)
            # Reset clears the baseline -> not ready. Enter "collecting" and let
            # the epoch loop publish "done" once it re-locks.
            self._calibrating = True
            self._baseline_was_ready = False
            self._publish_calibration(
                client,
                "Recalibrating... hold still 1-2 min",
                phase="collecting",
            )
            return

        if topic == In.CMD_FEEDBACK:
            self._handle_feedback(client, data, now)
            return

        if topic == In.CONTEXT:
            self._handle_context(data, now)
            return

        if topic == In.ACQUISITION:
            batch = _parse_acquisition(data)
            if batch is None:
                return
            # PPG 100 Hz now arrives in the batch (a single optical tracker). Feed
            # the high-resolution consumers (production morphology + valence FD).
            self._feed_highrate_ppg(batch)
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
                        "watch->server skew = %.1f s, delivery backlog (data arriving late)",
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
            if self.research.active:
                self._publish_legacy(client, batch, result)
            return

        return

    def _feed_highrate_ppg(self, batch) -> None:
        """Feed the high-resolution PPG consumers (100 Hz) from the batch.

        PPG-ul 100 Hz vine acum in payload-ul acquisition batch (un singur tracker
        optic pe ceas, nu mai e topic separat). Morfologia (productie, axa valenta)
        si motoarele de valenta FD (research) primesc forma de unda densa de aici.
        Daca batch-ul nu poarta PPG (raw PPG off), nu face nimic."""
        green = batch.ppg_green or []
        ts = batch.ppg_timestamps_ms or []
        if not green or len(green) != len(ts):
            return
        samples = [(int(t), int(g)) for t, g in zip(ts, green)]
        if not samples:
            return
        # Morphology is the only PRODUCTION consumer (internal 100->64 downsample).
        if self._morpho_clf is not None and self._morpho_clf.ready:
            self._morpho_clf.ingest_ondemand_ppg(samples)
        # The valence-FD engines (research, observed-only) consume the same wave.
        for name in ("_valence_fd", "_valence_wesad", "_valence_eevr", "_valence_case"):
            c = getattr(self.research, name, None)
            if c is not None:
                c.ingest_ondemand_ppg(samples)

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
        # epoch ticks (every 30 s), not on every 1 Hz batch, its median/hysteresis
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
            self._publish_emotion_state(client, result, batch)

        # Calibration progress: announce "done" the moment the baseline re-locks
        # after a recalibrate, so the watch stops the spinner.
        if self._calibrating and result.baseline_ready and not self._baseline_was_ready:
            self._calibrating = False
            self._publish_calibration(
                client, "Profile calibrated", phase="done"
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

    def _enrich_valence(self, out_dict, track, result) -> float | None:
        """Recenter one valence model's raw valence_z on the subject's personal
        resting baseline and add the calibrated fields to its published dict, the
        same way for WESAD, EEVR and CASE so all three show subject-relative emotion
        (neutral measured, not assumed 0). Returns the 60 s-smoothed personal valence
        (or None if no valence_z this epoch).

        `track` is the model's ValenceTrack (baseline + smoother + verdict
        stabiliser). It also folds this epoch into the track's stabiliser (confidence
        gate + median + hysteresis) and attaches the READY quadrant (emotion /
        emotion_code) to the dict, so the dashboard reads a stable verdict instead
        of recomputing it in SQL."""
        vz = out_dict.get("valence_z")
        if vz is None:
            return None
        vbase = track.baseline
        is_resting = (
            result is not None
            and getattr(result, "motion_state", "") == "still"
            and getattr(result, "signal_quality", 0.0) >= 0.5
        )
        if is_resting:
            vbase.observe_resting(float(vz), now=time.time())
        personal = vbase.valence_personal(float(vz))
        smoothed = track.smoother.update(personal, time.time())
        deadband = vbase.deadband(
            reactivity_scale=self.pipeline.reactivity.deadband_scale
        )
        out_dict["valence_personal"] = round(personal, 4)
        out_dict["valence_personal_60s"] = round(smoothed, 4)
        out_dict["valence_deadband"] = round(deadband, 4)
        out_dict["neutral_valence_z"] = round(vbase.neutral_valence_z or 0.0, 4)
        out_dict["valence_baseline_ready"] = vbase.is_ready

        # Stabilised 2D Russell quadrant for THIS model, the single source of
        # truth the dashboard displays (no CASE WHEN in SQL). Always emitted, not
        # gated on VR: valence here is the model's own personal-calibrated pulse
        # valence; arousal is the validated z. Honest at low confidence because
        # the stabiliser's gate pulls uncertain epochs to Neutru.
        arousal_z = self._current_arousal_z(result)
        conf = float(out_dict.get("confidence") or 0.0)
        label, code = track.stabilizer.update(
            arousal_z, smoothed, conf, deadband, vbase.is_ready,
        )
        out_dict["emotion"] = label
        out_dict["emotion_code"] = code
        return smoothed

    @staticmethod
    def _current_arousal_z(result) -> float:
        """Personal arousal z for this epoch (>=0 = at/above resting baseline)."""
        if result is not None and getattr(result, "decision", None) is not None:
            return float(getattr(result.decision,
                                 "stress_index_z_filtered", 0.0) or 0.0)
        return 0.0

    def _publish_legacy(self, client, batch, result) -> None:
        """Publish the parallel research engines on biofizic/legacy/* (never VR)."""
        out = self.research.run(batch=batch, result=result, baseline=self.pipeline.baseline)
        ts = self._anchor_ms
        log.debug("legacy out: ppg=%s resp=%s",
                  out.ppg is not None, out.respiration)
        if out.ppg is not None:
            client.publish("biofizic/legacy/ppg", json.dumps({"ts": ts, **out.ppg}), qos=0)
        if out.respiration is not None:
            client.publish("biofizic/legacy/resp", json.dumps({"ts": ts, **out.respiration}), qos=0)
        if out.valence_fd is not None:
            client.publish("biofizic/legacy/valence_fd", json.dumps({"ts": ts, **out.valence_fd}), qos=0)
        if out.valence_wesad is not None:
            # Personal valence calibration: recenter the cross-device WESAD model
            # on this subject (neutral measured on resting epochs, not assumed 0).
            wesad_track = self.research.valence_tracks["wesad"]
            vbase = wesad_track.baseline
            smoothed = self._enrich_valence(out.valence_wesad, wesad_track, result)
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
                    # If Unity sent a FRESH scene context (<10 s), the valence axis
                    # comes from the SCENE (honest, strong) instead of the weak
                    # PPG valence. Arousal still comes from physiology. Otherwise
                    # fall back to the PPG valence as before.
                    ctx = self._last_context
                    use_context = (
                        ctx is not None and (time.time() - ctx["ts"]) < 10.0
                    )
                    # Intensity gate (empirically grounded): PPG valence is only
                    # reliable (~87% LOSO on WESAD) when the autonomic reaction is
                    # STRONG. On mild stimuli (small |arousal_z|) place/dislike
                    # collapses to chance (~51% across CASE/EMOGNITION/DEAP), so we
                    # must not assert a valence sign there. Below the threshold the
                    # PPG valence is suppressed (treated as "undetermined") and only
                    # arousal is reported. The VR scene context is exempt, it is an
                    # honest external prior, not the weak PPG signal.
                    ppg_intensity_ok = abs(arousal_z) >= VALENCE_INTENSITY_GATE_Z
                    if use_context:
                        valence_for_verdict = ctx["valence_prior"]
                        conf_for_verdict = 1.0
                        deadband_for_verdict = 0.15
                        self._valence_source = "context"
                    elif ppg_intensity_ok:
                        valence_for_verdict = smoothed
                        conf_for_verdict = self._last_valence_confidence
                        deadband_for_verdict = out.valence_wesad["valence_deadband"]
                        self._valence_source = "ppg"
                    else:
                        # Reaction too mild for trustworthy valence -> undetermined.
                        # valence_for_verdict=0 keeps the verdict on the arousal axis
                        # only (neutral valence), source flags the gate for the board.
                        valence_for_verdict = 0.0
                        conf_for_verdict = 0.0
                        deadband_for_verdict = out.valence_wesad["valence_deadband"]
                        self._valence_source = "undetermined_low_intensity"
                    self._last_emotion_verdict = self._valence_stabilizer.update(
                        arousal_z, valence_for_verdict, conf_for_verdict,
                        deadband_for_verdict, vbase.is_ready or use_context,
                    )
            client.publish("biofizic/legacy/valence_wesad", json.dumps({"ts": ts, **out.valence_wesad}), qos=0)

        # EEVR / CASE valence: observed-only comparison models, each personally
        # recentered on its OWN baseline (same as WESAD) so the dashboard shows
        # calibrated emotion for all three, not raw valence. They do NOT feed the
        # watch verdict (no stabiliser / decision cache).
        if out.valence_eevr is not None:
            self._enrich_valence(
                out.valence_eevr, self.research.valence_tracks["eevr"], result)
            client.publish("biofizic/legacy/valence_eevr", json.dumps({"ts": ts, **out.valence_eevr}), qos=0)
        if out.valence_case is not None:
            self._enrich_valence(
                out.valence_case, self.research.valence_tracks["case"], result)
            client.publish("biofizic/legacy/valence_case", json.dumps({"ts": ts, **out.valence_case}), qos=0)

        # Cache the three models' predictions + the live arousal z at this instant,
        # so a feedback tap can pair the user's label with what each model said
        # right then (the model-vs-truth validation). Cheap, overwritten each batch.
        self._last_pred = {
            "wesad": (out.valence_wesad or {}).get("p_positive"),
            "eevr": (out.valence_eevr or {}).get("p_positive"),
            "case": (out.valence_case or {}).get("p_positive"),
            "arousal_z": (
                result.decision.stress_index_z_filtered
                if result is not None and result.decision is not None else None
            ),
            "ts": int(ts),
        }

    def _handle_feedback(self, client, data, now) -> None:
        """A feedback tap: pair the user's chosen state with the live model feature
        vectors + what the models predicted, append to data/feedback_labels.jsonl,
        and try a personal retrain once enough labels exist. Re-publish a summary."""
        from affectus.dsp.feedback_store import append_feedback, normalize_state

        quadrant = data.get("quadrant")
        # Accept the new 3 states and the old 4 quadrants (mapped). None => unknown.
        if normalize_state(quadrant) is None:
            log.warning("feedback ignored: unknown label %r", quadrant)
            return
        eng = getattr(self.research, "_valence_wesad", None)
        features = getattr(eng, "last_features", None) if eng is not None else None
        feat_ts = getattr(eng, "last_features_ts", 0) if eng is not None else 0
        # The live model feature vectors at label time. THESE feed the retrain loop.
        state_features = getattr(self._state_clf, "last_features", None) if self._state_clf else None
        morpho_features = getattr(self._morpho_clf, "last_features", None) if self._morpho_clf else None
        # Feature freshness: how old is the labelled window vs the tap.
        tap_ms = data.get("ts") if isinstance(data.get("ts"), (int, float)) else now * 1000
        age_s = (float(tap_ms) - feat_ts) / 1000.0 if feat_ts else None
        preds = getattr(self, "_last_pred", {}) or {}
        row = append_feedback(
            quadrant, features,
            arousal_z=preds.get("arousal_z"),
            features_age_s=age_s,
            preds=preds,
            state_features=state_features,
            morpho_features=morpho_features,
        )
        log.info("Feedback stored: %s -> %s (state_feats=%s)",
                 quadrant, row["state"], state_features is not None)
        # After storing, try a personal retrain (cheap, user-paced).
        self._maybe_retrain_personal(client)
        # Summary for Influx/Grafana (NOT the 33 features, those stay in the JSONL).
        client.publish("biofizic/legacy/feedback", json.dumps({
            "ts": row["ts"],
            "quadrant_code": row["quadrant_code"],
            "arousal_z": row["arousal_z"],
            "wesad_p": row["wesad_p_positive"],
            "eevr_p": row["eevr_p_positive"],
            "case_p": row["case_p_positive"],
        }), qos=0)

    def _maybe_retrain_personal(self, client) -> None:
        """After a label is stored, try to retrain a personal 3-state model and,
        if it passed the self-check, hot-reload it into the live classifier and
        announce it on the dashboard. Triggered only on taps (user-paced), so the
        RandomForest fit here is cheap enough to run inline. Never fatal."""
        if self._state_clf is None:
            return
        try:
            from affectus.research.stari.personal_trainer import maybe_retrain
            res = maybe_retrain()
        except Exception:
            log.exception("personal retrain failed")
            return
        if not res.get("trained"):
            log.info("Personal model not (re)trained: %s | counts=%s",
                     res.get("reason"), res.get("counts"))
            return
        became_personal = self._state_clf.reload()
        log.info("Personal model active (self_check=%s, counts=%s)",
                 res.get("self_check"), res.get("counts"))
        sc = res.get("self_check") or {}
        client.publish("biofizic/legacy/personal_model", json.dumps({
            "ts": int(time.time() * 1000),
            "active": 1 if became_personal else 0,
            "personal_acc": sc.get("personal"),
            "generic_acc": sc.get("generic"),
        }), qos=0)

    def _handle_context(self, data, now) -> None:
        """VR scene context from Unity: cache the raw visual cues + the derived
        valence prior, timestamped, so the emotion verdict can use the SCENE's
        valence (honest) when it is fresh. Degenerate input -> ignored."""
        from affectus.dsp.scene_context import (
            context_to_valence_prior, valence_prior_label,
        )

        def _f(key, default=0.5):
            v = data.get(key)
            return float(v) if isinstance(v, (int, float)) else default

        light = _f("light"); r = _f("r"); g = _f("g"); b = _f("b"); motion = _f("motion", 0.0)
        prior = context_to_valence_prior(light, r, g, b, motion)
        self._last_context = {
            "scene": str(data.get("scene") or "unknown"),
            "light": round(light, 3), "r": round(r, 3), "g": round(g, 3), "b": round(b, 3),
            "motion": round(motion, 3),
            "valence_prior": prior,
            "label": valence_prior_label(prior),
            "ts": now,
        }

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
            # NOTE: valence + 2D emotion verdict are NOT sent to the watch, they
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
        # 2D Russell emotion verdict (arousal × valence). Valence comes from the VR
        # scene CONTEXT when fresh (honest: valence_source="context"), not from the
        # pulse. Only attached when a context-derived verdict exists, so the watch's
        # arousal-only path is unchanged when there's no VR running.
        ctx = self._last_context
        if ctx is not None and (time.time() - ctx["ts"]) < 10.0:
            label, code = self._last_emotion_verdict
            payload["emotion_2d"] = label
            payload["emotion_2d_code"] = code
            payload["valence_prior"] = ctx["valence_prior"]
            payload["valence_source"] = "context"
            payload["scene"] = ctx["scene"]
        if result is not None:
            # Report the window that ACTUALLY drove the verdict (w60 normally,
            # w30 only during the first 60 s before w60 is computable), not a
            # hardcoded label, the windows dashboard must not misattribute the
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
        client.publish(Out.LIVE, json.dumps(payload), qos=0)

    def _apply_live_arousal_smoothing(
        self, payload: dict, decision: PhysiologyDecision
    ) -> None:
        """
        Apply streak-based hysteresis to the integer arousal_10 reported on
        biofizic/out/live.

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
        client.publish(Out.WINDOWS, json.dumps(payload), qos=0)

    def _publish_state(
        self,
        client,
        decision: PhysiologyDecision,
        *,
        result: MultiWindowResult | None = None,
    ) -> None:
        """
        Publish the epoch decision (every 30 s) on biofizic/out/arousal. The message
        is retained so a watch reconnecting between epochs can bootstrap from
        the last known decision without waiting up to 30 s for the next one.
        """
        payload = self._decision_payload(decision, live=False, result=result)
        client.publish(Out.AROUSAL, json.dumps(payload), qos=1, retain=True)

    def _publish_emotion_state(self, client, result: MultiWindowResult, batch) -> None:
        """3-state emotion verdict (CALM/DISCONFORT/PLACUT) on biofizic/out/emotion.
        Research/viz only, NOT the production arousal verdict. Adapts to the user via a
        personal rest baseline: feeds calm windows, classifies once the baseline locks
        (~12 rest windows). Returns nothing until then (no premature verdict)."""
        if self._state_clf is None or not self._state_clf.ready:
            return
        decision = result.decision
        if decision is None:
            return
        window = result.best  # WindowResult care a decis (w60 normal, w30 la start)
        if window is None:
            return
        # Feed the personal baseline from stable, clean windows: still + decent
        # signal quality. NOT gated on `alert`, the CUSUM alert tracks arousal
        # CHANGE, not whether the user is at rest, and it stays on for tired/active
        # users, which would starve the baseline forever (it never locks).
        is_rest = (decision.motion_state == "still"
                   and decision.signal_quality >= 0.5)
        if is_rest:
            self._state_clf.observe_rest(window, batch)
            if self._morpho_clf is not None and self._morpho_clf.ready:
                self._morpho_clf.observe_rest()
        out = self._state_clf.predict(window, batch)
        morpho = self._morpho_clf.predict() if (self._morpho_clf is not None
                                                and self._morpho_clf.ready) else None
        if out is None:
            # Baseline not locked yet, tell the dashboard we're still calibrating.
            payload = {"ts": self._anchor_ms, "state": "CALIBRARE",
                       "confidence": 0.0, "p_calm": 0.0, "p_disconfort": 0.0,
                       "p_placut": 0.0, "baseline_ready": False, "calibrating": True}
        else:
            probs = out["probs"]
            arousal = int(decision.display_arousal_10)
            # Valenta (X) din morfologie cand e gata (separa disc/plac la 86%), altfel HRV.
            # Daca morfologia voteaza CALM, valenta NU are directie clara -> 0 (centru),
            # ca CALM-ul sa nu produca un semn fals de valenta. Doar cand morfologia
            # inclina spre disconfort SAU placut, valence_x reflecta directia (normalizat).
            mp = morpho["probs"] if morpho else probs
            morpho_says_calm = morpho is not None and morpho["state"] == "CALM"
            if morpho_says_calm:
                valence_x = 0.0
            else:
                p_disc = float(mp.get("DISCONFORT", 0.0))
                p_plac = float(mp.get("PLACUT", 0.0))
                denom = p_disc + p_plac
                valence_x = (p_plac - p_disc) / denom if denom > 1e-6 else 0.0
            quadrant, emotions, source, final_conf, valence_reliable = self._russell_quadrant(
                arousal, valence_x, morpho)
            final_state = quadrant
            payload = {"ts": self._anchor_ms, "state": final_state,
                       "emotions": emotions, "confidence": final_conf, "source": source,
                       "valence_x": round(valence_x, 3),
                       "arousal_y": float(decision.display_arousal_10),
                       "valence_reliable": valence_reliable,
                       "p_calm": probs.get("CALM", 0.0),
                       "p_disconfort": probs.get("DISCONFORT", 0.0),
                       "p_placut": probs.get("PLACUT", 0.0),
                       "morpho_state": morpho["state"] if morpho else None,
                       "morpho_conf": morpho["confidence"] if morpho else 0.0,
                       "baseline_ready": True, "calibrating": False}
        # Retained: a watch (re)subscribing mid-session gets the last verdict
        # immediately instead of waiting for the next epoch (the verdict is QoS 0
        # and would otherwise be lost if the watch reconnected between publishes).
        client.publish(Out.EMOTION, json.dumps(payload), qos=0, retain=True)

    @staticmethod
    def _russell_quadrant(arousal: int, valence_x: float, morpho) -> tuple:
        """Mapeaza arousal(0-10) x valenta(-1..+1) -> o stare cu nume uman + sugestii.

        Structura bazata pe ce se separa ONEST din date:
          - Arousal se separa binar (jos/sus) la 68-73% pe senzori reali -> 3 niveluri
            native de activare: RELAXAT (jos) / NORMAL (baseline) / ACTIVAT (sus).
          - Valenta (placut/neplacut) NU se separa din HRV; doar din MORFOLOGIE si DOAR
            la arousal mare (morfologia separa 83% disc-vs-placut pe semnal bun, validat).
        Deci FUZIUNEA cu valenta intra DOAR sus + morfologie fiabila -> BUCUROS/STRESAT.
        Sub acel prag, starea vine pur din nivelul de arousal (validat), fara a ghici
        polaritatea. Asa userul vede mereu o emotie reala, iar sistemul ramane onest."""
        VAL_NEUTRAL = 0.12   # |valence| sub asta = neutru pe valenta
        AROUSAL_LO, AROUSAL_HI = 4, 6  # <=4 jos (RELAXAT), 5-6 normal, >=7 sus (ACTIVAT)
        # valenta e fiabila DOAR daca morfologia e gata SI arousalul e mare
        valence_reliable = (morpho is not None and arousal > AROUSAL_HI)
        conf = round(float(morpho["confidence"]) if (morpho and valence_reliable)
                     else 0.5, 2)

        if arousal > AROUSAL_HI:  # SUS (ACTIVAT)
            # Daca valenta e fiabila (morfologie gata) si clara, putem numi exact
            # starea. Altfel NU ghicim placut-vs-neplacut: spunem ZONA = ambele stari
            # care se confunda la activare mare ("Stres sau Entuziasm"), onest. Userul
            # stie care e (el o simte); sistemul nu pretinde ce nu poate masura.
            if valence_reliable and valence_x >= VAL_NEUTRAL:
                return ("Entuziasmat", ["Bucuros", "Energic", "Vesel"],
                        "morfologie", conf, True)
            if valence_reliable and valence_x <= -VAL_NEUTRAL:
                return ("Stresat", ["Tensionat", "Iritat", "Agitat"],
                        "morfologie", conf, True)
            # activat dar valenta incerta -> ZONA, nu o eticheta falsa. Cele doua stari
            # tipice la activare mare care nu se pot separa din puls.
            return ("Stres sau Entuziasm", ["esti puternic activat", "energie ridicata"],
                    "arousal", conf, False)
        if arousal <= AROUSAL_LO:  # JOS (dezactivat) - stare nativa de arousal, validata
            return ("Relaxat", ["Linistit", "Calm", "Odihnit"],
                    "arousal", round(1.0 - arousal / AROUSAL_LO, 2), False)
        # MIJLOC (5-6) - la baseline-ul personal
        return ("Normal", ["Echilibrat", "Concentrat", "Atent"],
                "arousal", 0.6, False)

    def _publish_calibration(self, client, message: str, *, phase: str = "done") -> None:
        payload = {
            "ts": int(time.time() * 1000),
            "action": "profile",
            "phase": phase,
            "message": message,
            "profile_ready": self.pipeline.baseline.is_ready,
        }
        client.publish(Out.CALIBRATION, json.dumps(payload), qos=1, retain=True)


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
