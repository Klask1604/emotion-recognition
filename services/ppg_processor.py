#!/usr/bin/env python3
"""
PPG raw processing pipeline for Galaxy Watch 7.
Runs in parallel with compute-engine; publishes biofizic/ppg_hrv (RMSSD + z_pulse_amp).

Usage:
    python ppg_processor.py
    python ppg_processor.py --broker paxbespoke.automateflow.ro --port 1883
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
import time
from collections import deque

import numpy as np
import paho.mqtt.client as mqtt
from scipy import signal

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ppg_proc")

# ── Constante ──────────────────────────────────────────────────────────────────
FS           = 25        # Hz — frecventa nominala GW7 PPG
WINDOW_SEC   = 30        # secunde de PPG per epoca HRV
MIN_PEAKS    = 6         # minim batai curate (30s @ ~85bpm ≈ 40; 6 = RMSSD minim valid)
BUFFER_SEC   = 120       # buffer circular (4 ferestre)

LOW_HZ       = 0.5
HIGH_HZ      = 4.0
BUTTER_ORDER = 4

# 0.38s = ~157 bpm max — suficient pentru efort moderat, previne double peaks
MIN_PEAK_DIST_SAMPLES = int(FS * 0.38)

# Limite fiziologice pentru artefact guard
HR_MIN   = 35    # bpm — sub asta = ceas dezlipit sau artefact
HR_MAX   = 160   # bpm — peste asta = artefact motion
RMSSD_MAX = 220  # ms  — RMSSD > 220ms = artefact aproape sigur

# Ingest: ignoră retain vechi / resetează la pauză lungă
MAX_TS_AGE_MS      = 120_000   # 2 min — batch retain de ieri
MAX_INGEST_GAP_MS  = 15_000    # 15s — Samsung face pauze naturale 8-9s la ecran oprit
MAX_WINDOW_GAP_MS  = 12_000    # 12s — trim prefix dupa gap (sub pauzele Samsung)
MAX_REAL_GAP_MS    = 15_000    # 15s — peste asta = ceas dezlipit / sesiune moarta

# Prag miscare pentru PPG HRV — acc_rms la incheietura e ~0.6–1.2 in repaus (HAND/SCROLL).
# Sub 2.0 m/s^2 inca putem incerca peak detection; peste = locomotion / artefact major.
PPG_MOTION_ACC_BLOCK = 2.0
PPG_BLOCKED_MODES = frozenset({"LOCOMOTION"})


# ── Filtru Butterworth ─────────────────────────────────────────────────────────
def _build_sos(low: float, high: float, fs: int, order: int) -> np.ndarray:
    nyq = fs / 2.0
    return signal.butter(order, [low / nyq, high / nyq], btype="band", output="sos")


SOS = _build_sos(LOW_HZ, HIGH_HZ, FS, BUTTER_ORDER)


def apply_filter(samples: np.ndarray) -> np.ndarray:
    if len(samples) < FS * 4:
        return samples
    return signal.sosfiltfilt(SOS, samples)


def preprocess_ppg(green: np.ndarray) -> np.ndarray:
    """Detrend + scale — Elgendi/scipy pe valorile Samsung (~10k) esueaza fara asta."""
    x = green.astype(float) - float(np.median(green))
    std = float(np.std(x))
    if std > 1e-6:
        x = x / std
    return x


def split_contiguous(ts_ms: list[int], max_gap_ms: int = 2500) -> list[tuple[int, int]]:
    """Indecsi [start, end] pentru segmente fara pauze mari (batching Samsung)."""
    if not ts_ms:
        return []
    segments: list[tuple[int, int]] = []
    start = 0
    for i in range(1, len(ts_ms)):
        if ts_ms[i] - ts_ms[i - 1] > max_gap_ms:
            segments.append((start, i - 1))
            start = i
    segments.append((start, len(ts_ms) - 1))
    return segments


def resample_uniform(green: np.ndarray, ts_ms: list[int], fs: int) -> np.ndarray | None:
    """Reesantioneaza pe grila uniforma folosind timestamps reale (nu presupune 25 Hz fix)."""
    if len(green) < 4 or len(ts_ms) != len(green):
        return None
    t_s = (np.array(ts_ms, dtype=float) - ts_ms[0]) / 1000.0
    dur = float(t_s[-1])
    if dur < 5.0:
        return None
    n_out = max(int(dur * fs), fs * 5)
    t_uniform = np.linspace(0.0, dur, n_out)
    return np.interp(t_uniform, t_s, green.astype(float))


def _score_peaks(peaks: np.ndarray, fs: int) -> float:
    if len(peaks) < 3:
        return float(len(peaks))
    ibis = np.diff(peaks) / fs * 1000.0
    valid = ibis[(ibis >= 400) & (ibis <= 1600)]
    if len(valid) == 0:
        return float(len(peaks)) * 0.1
    med = float(np.median(valid))
    hr_penalty = min(abs(med - 800.0) / 800.0, 1.0)
    return len(valid) * (1.0 - 0.5 * hr_penalty)


def _find_peaks_robust(ppg: np.ndarray, fs: int) -> np.ndarray:
    """Incearca scipy, scipy pe semnal inversat si Elgendi — alege cel mai plauzibil."""
    min_dist = MIN_PEAK_DIST_SAMPLES
    prom = max(0.008 * float(np.ptp(ppg)), 1e-6)
    candidates: list[tuple[str, np.ndarray]] = []

    p_pos, _ = signal.find_peaks(ppg, distance=min_dist, prominence=prom)
    candidates.append(("scipy", _dedup_peaks(p_pos, ppg, min_dist)))

    p_neg, _ = signal.find_peaks(-ppg, distance=min_dist, prominence=prom)
    candidates.append(("scipy_inv", _dedup_peaks(p_neg, ppg, min_dist)))

    candidates.append(("elgendi", _find_ppg_peaks_nk(ppg, fs)))

    best_name, best_peaks = candidates[0]
    best_score = -1.0
    for name, peaks in candidates:
        score = _score_peaks(peaks, fs)
        if score > best_score:
            best_score = score
            best_peaks = peaks
            best_name = name
    log.debug("Peak method=%s n=%d score=%.1f", best_name, len(best_peaks), best_score)
    return best_peaks


# ── Peak detection fără NeuroKit2 (fallback robust) ───────────────────────────
def _find_ppg_peaks(ppg: np.ndarray, fs: int) -> np.ndarray:
    """
    Detectie vârfuri sistolice PPG cu scipy.signal.find_peaks.
    Minimum distanta = 0.5s (= 120 bpm max).
    """
    min_dist = MIN_PEAK_DIST_SAMPLES
    prom = max(0.02 * float(np.ptp(ppg)), 1e-6)
    peaks, _ = signal.find_peaks(ppg, distance=min_dist, prominence=prom)
    return peaks


def _dedup_peaks(peaks: np.ndarray, ppg: np.ndarray, min_dist: int = MIN_PEAK_DIST_SAMPLES) -> np.ndarray:
    """
    Elimina double peaks (notch dicrot detectat ca vârf sistolic).
    Dacă doua vârfuri sunt mai apropiate de min_dist samples, pastreaza pe cel mai înalt.
    """
    if len(peaks) < 2:
        return peaks
    result = [int(peaks[0])]
    for pk in peaks[1:]:
        pk = int(pk)
        if pk - result[-1] < min_dist:
            if ppg[pk] > ppg[result[-1]]:
                result[-1] = pk
        else:
            result.append(pk)
    return np.array(result, dtype=int)


def _find_ppg_peaks_nk(ppg: np.ndarray, fs: int) -> np.ndarray:
    """Peak detection cu NeuroKit2 + deduplicare anti-notch."""
    min_dist = MIN_PEAK_DIST_SAMPLES
    peakwindow = min_dist / fs  # aliniaza distanta minima Elgendi cu dedup-ul
    try:
        import neurokit2 as nk  # noqa: PLC0415
        info = nk.ppg_findpeaks(
            ppg, sampling_rate=fs, method="elgendi", peakwindow=peakwindow,
        )
        peaks = np.array(info["PPG_Peaks"], dtype=int)
    except Exception:
        peaks = _find_ppg_peaks(ppg, fs)
    return _dedup_peaks(peaks, ppg, min_dist)


# ── Features HRV + amplitudine din PPG filtrat ────────────────────────────────
def extract_hrv_features(
    ppg_filtered: np.ndarray, fs: int, ts_ms: list[int] | None = None,
) -> tuple[dict | None, str]:
    peaks = _find_peaks_robust(ppg_filtered, fs)

    if len(peaks) < MIN_PEAKS:
        return None, f"peaks={len(peaks)}<{MIN_PEAKS}"

    if ts_ms is not None and len(ts_ms) == len(ppg_filtered):
        ibi_ms_raw = [
            float(ts_ms[int(peaks[i + 1])] - ts_ms[int(peaks[i])])
            for i in range(len(peaks) - 1)
        ]
    else:
        ibi_ms_raw = (np.diff(peaks) / fs * 1000.0).tolist()

    ibi_valid = [x for x in ibi_ms_raw if 300 <= x <= 2000]
    if len(ibi_valid) < MIN_PEAKS - 1:
        return None, f"ibi_valid={len(ibi_valid)}<{MIN_PEAKS - 1}"

    med = float(np.median(ibi_valid))
    ibi_clean = [x for x in ibi_valid if abs(x - med) <= 0.20 * med]
    if len(ibi_clean) < MIN_PEAKS:
        return None, f"ibi_clean={len(ibi_clean)}<{MIN_PEAKS}"

    diffs    = np.diff(ibi_clean)
    rmssd    = float(np.sqrt(np.mean(diffs ** 2)))
    mean_ibi = float(np.mean(ibi_clean))
    mean_hr  = 60000.0 / mean_ibi if mean_ibi > 0 else 0.0
    sdnn     = float(np.std(ibi_clean, ddof=0))
    pnn50    = float(100.0 * np.sum(np.abs(diffs) > 50) / len(diffs)) if len(diffs) > 0 else 0.0

    if not (HR_MIN <= mean_hr <= HR_MAX):
        return None, f"hr={mean_hr:.0f} out of range"
    if rmssd > RMSSD_MAX:
        return None, f"rmssd={rmssd:.0f}>{RMSSD_MAX}"

    amplitudes = []
    for i, pk in enumerate(peaks[:-1]):
        seg = ppg_filtered[pk : peaks[i + 1]]
        if len(seg) > 2:
            amp = float(ppg_filtered[pk]) - float(np.min(seg))
            if amp > 0:
                amplitudes.append(amp)

    pulse_amp_mean = float(np.mean(amplitudes)) if amplitudes else 0.0
    pulse_amp_std  = float(np.std(amplitudes))  if amplitudes else 0.0

    return {
        "ibi_n":          len(ibi_clean),
        "mean_hr":        round(mean_hr, 1),
        "mean_ibi_ms":    round(mean_ibi, 1),
        "rmssd":          round(rmssd, 2),
        "sdnn":           round(sdnn, 2),
        "pnn50":          round(pnn50, 1),
        "pulse_amp_mean": round(pulse_amp_mean, 4),
        "pulse_amp_std":  round(pulse_amp_std, 4),
        "ibi_ms":         [round(x, 1) for x in ibi_clean],
        "peak_count":     int(len(peaks)),
    }, "ok"


# ── Buffer circular PPG cu timestamps ─────────────────────────────────────────
class PpgBuffer:
    def __init__(self, fs: int, buffer_sec: int) -> None:
        self.fs = fs
        maxn = fs * buffer_sec
        self._green: deque[int] = deque(maxlen=maxn)
        self._ts_ms: deque[int] = deque(maxlen=maxn)

    def clear(self) -> None:
        self._green.clear()
        self._ts_ms.clear()

    def ingest(self, ts_list: list, green_list: list) -> int:
        """Adaugă samples; ignoră timestamps vechi. Returnează câte au intrat."""
        now_ms = int(time.time() * 1000)
        added = 0
        for ts, g in zip(ts_list, green_list):
            ts = int(ts)
            if ts < now_ms - MAX_TS_AGE_MS:
                continue
            if self._ts_ms:
                delta = ts - self._ts_ms[-1]
                if delta > MAX_INGEST_GAP_MS:
                    self.clear()
                    log.warning(
                        "PPG buffer reset la ingest (pauză %.1fs)",
                        delta / 1000.0,
                    )
                elif delta <= 0:
                    continue
            self._ts_ms.append(ts)
            self._green.append(int(g))
            added += 1
        return added

    def get_window(self, window_sec: int) -> tuple[np.ndarray, list[int]]:
        """Ultimele `window_sec` secunde după timestamp, nu după număr de samples."""
        if not self._ts_ms:
            return np.array([]), []
        cutoff = self._ts_ms[-1] - window_sec * 1000
        ts_all = list(self._ts_ms)
        green_all = list(self._green)
        start = 0
        for i, t in enumerate(ts_all):
            if t >= cutoff:
                start = i
                break
        ts = ts_all[start:]
        green = np.array(green_all[start:], dtype=float)
        return green, ts

    @property
    def total_sec(self) -> float:
        if len(self._ts_ms) < 2:
            return 0.0
        return (self._ts_ms[-1] - self._ts_ms[0]) / 1000.0

    @property
    def span_sec(self) -> float:
        """Durata ultimelor 30s de date (pentru decizia de procesare)."""
        if len(self._ts_ms) < 2:
            return 0.0
        cutoff = self._ts_ms[-1] - WINDOW_SEC * 1000
        first = next((t for t in self._ts_ms if t >= cutoff), self._ts_ms[0])
        return (self._ts_ms[-1] - first) / 1000.0

    @property
    def sample_count(self) -> int:
        return len(self._green)


# ── Baseline amplitudine puls personalizat ────────────────────────────────────
class AmplitudeBaseline:
    def __init__(self) -> None:
        self.rest_amp_mean: float | None = None
        self.rest_amp_std:  float = 1.0
        self._buffer: list[float] = []

    def observe(self, amp: float, acc_rms: float) -> None:
        # Amplitudine din semnal filtrat normalizat (~0.05–0.5), nu ADC brut
        if amp <= 0.001:
            return
        if amp > 5.0:
            return
        # Outlier absolut fata de median global — previne drift baseline
        if len(self._buffer) >= 10:
            global_med = float(np.median(self._buffer))
            if amp > 2.5 * global_med:
                log.debug("Outlier absolut: %.0f vs global_med=%.0f", amp, global_med)
                return
        # Verificare outilier fata de buffer recent (fallback)
        if self._buffer:
            local_med = float(np.median(self._buffer[-20:])) if len(self._buffer) >= 5 else amp
            if amp > 3.0 * local_med and local_med > 0.01:
                log.debug("Amplitudine outlier ignorata: %.4f (median=%.4f)", amp, local_med)
                return
        if acc_rms >= 1.2:
            return
        self._buffer.append(amp)
        if len(self._buffer) > 200:
            self._buffer = self._buffer[-100:]
        # Necesita 15 observatii (era 20) pentru initializare mai rapida
        if len(self._buffer) >= 15:
            arr = np.array(self._buffer[-40:])
            # Foloseste median robust in loc de mean pentru std
            self.rest_amp_mean = float(np.median(arr))
            # IQR-based std pentru robustete la outlieri
            q75, q25 = float(np.percentile(arr, 75)), float(np.percentile(arr, 25))
            self.rest_amp_std  = max(0.01, (q75 - q25) * 0.7413)
            log.info(
                "Baseline amplitudine actualizat: mean=%.0f std=%.0f (n=%d)",
                self.rest_amp_mean, self.rest_amp_std, len(arr),
            )

    def z_score(self, amp: float) -> float:
        """Z pozitiv = vasoconstrictie = activare simpatică."""
        if self.rest_amp_mean is None:
            return 0.0
        return float(np.clip(
            (self.rest_amp_mean - amp) / self.rest_amp_std,
            -3.0, 3.0,
        ))

    @property
    def ready(self) -> bool:
        return self.rest_amp_mean is not None


def trim_after_gaps(
    green: np.ndarray, ts_list: list[int], max_gap_ms: int = MAX_WINDOW_GAP_MS,
) -> tuple[np.ndarray, list[int], float]:
    """Taie prefixul până după ultimul gap mare — returnează și max_gap_ms."""
    if len(ts_list) < 2:
        return green, ts_list, 0.0
    gaps = np.diff(np.array(ts_list))
    max_gap = float(np.max(gaps))
    if max_gap <= max_gap_ms:
        return green, ts_list, max_gap
    cut = 0
    for i, g in enumerate(gaps):
        if g > max_gap_ms:
            cut = i + 1
    return green[cut:], ts_list[cut:], max_gap


# ── Engine principal ───────────────────────────────────────────────────────────
class PpgEngine:
    def __init__(self, broker: str, port: int) -> None:
        self.buf      = PpgBuffer(FS, BUFFER_SEC)
        self.baseline = AmplitudeBaseline()
        self._last_epoch_ts = 0.0
        self._epoch_count   = 0
        self._last_acc_rms  = 0.0
        self._sensor_acc_rms = 0.0
        self._sensor_acc_p90 = 0.0
        self._activity_mode = "UNKNOWN"
        self._last_green_mean = 0.0
        self._last_batch_samples = 0
        self._batches_recv  = 0
        self._last_batch_log = 0.0
        self._last_pipeline_pub = 0.0

        self.client = mqtt.Client(
            client_id="biofizic_ppg_proc",
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
        )
        self.client.on_connect = self._on_connect
        self.client.on_message = self._on_message
        self.broker = broker
        self.port   = port

    def start(self) -> None:
        self.client.connect(self.broker, self.port, keepalive=60)
        self.client.loop_forever()

    def _on_connect(self, client, userdata, flags, rc, props=None) -> None:
        if rc == 0:
            self.buf.clear()
            self._last_epoch_ts = time.time()
            log.info("PPG processor conectat la %s:%d (buffer golit)", self.broker, self.port)
            client.subscribe("biofizic/acquisition/batch", qos=0)
            client.subscribe("biofizic/ppg/raw", qos=0)
            client.subscribe("biofizic/ppg/batch", qos=0)
            client.subscribe("biofizic/sensors/batch", qos=0)
            client.subscribe("biofizic/state/live", qos=0)
        else:
            log.error("MQTT connect esuata rc=%d", rc)

    def _on_message(self, client, userdata, msg) -> None:
        try:
            data = json.loads(msg.payload.decode("utf-8", errors="replace"))
        except Exception:
            return

        if msg.topic == "biofizic/ppg/raw":
            if msg.retain:
                log.info("Ignorat batch PPG retinut (stale de la sesiunea anterioara)")
                return
            self._ingest_ppg(data)
        elif msg.topic == "biofizic/acquisition/batch":
            if int(data.get("schema", 0)) < 2:
                return
            ppg = data.get("ppg") or {}
            self._ingest_ppg_batch(
                {
                    "ts_ms": ppg.get("ts_ms") or [],
                    "green": ppg.get("green") or [],
                }
            )
            motion = data.get("motion") or {}
            acc_rms = float(motion.get("acc_rms", 0) or 0)
            if acc_rms > 0:
                self._sensor_acc_rms = acc_rms
                self._sensor_acc_p90 = float(
                    motion.get("acc_p90", self._sensor_acc_rms) or 0
                )
        elif msg.topic == "biofizic/ppg/batch":
            self._ingest_ppg_batch(data)
        elif msg.topic == "biofizic/sensors/batch":
            self._sensor_acc_rms = float(data.get("acc_rms", 0) or 0)
            self._sensor_acc_p90 = float(data.get("acc_p90", self._sensor_acc_rms) or 0)
        elif msg.topic == "biofizic/state/live":
            self._activity_mode = str(data.get("activity_mode", "UNKNOWN"))

    def _ingest_ppg_batch(self, data: dict) -> None:
        ts_list = data.get("ts_ms") or data.get("ts") or []
        green_list = data.get("green") or []
        if not ts_list or not green_list:
            return
        if isinstance(ts_list, int):
            ts_list = [ts_list]
        added = self.buf.ingest(ts_list, green_list)
        if added == 0:
            return
        self._batches_recv += 1
        self._last_batch_samples = added
        try:
            self._last_green_mean = float(np.mean([int(x) for x in green_list]))
        except Exception:
            pass
        self._publish_pipeline_status(epoch_skipped=False, skip_reason="ingest")
        now = time.time()
        if (
            (now - self._last_epoch_ts) >= WINDOW_SEC
            and self.buf.span_sec >= WINDOW_SEC * 0.85
        ):
            self._process_epoch(now)

    def _ingest_ppg(self, data: dict) -> None:
        ts_list    = data.get("ts",    [])
        green_list = data.get("green", [])
        if not ts_list or not green_list:
            return

        added = self.buf.ingest(ts_list, green_list)
        if added == 0:
            return
        self._batches_recv += 1
        self._last_batch_samples = added
        try:
            self._last_green_mean = float(np.mean(green_list))
        except Exception:
            pass
        self._publish_pipeline_status(epoch_skipped=False, skip_reason="ingest")

        now = time.time()
        if now - self._last_batch_log >= 30.0:
            self._last_batch_log = now
            log.info(
                "PPG buffer: %.1fs span, %d samples, %d batch-uri",
                self.buf.span_sec, self.buf.sample_count, self._batches_recv,
            )

        if (
            (now - self._last_epoch_ts) >= WINDOW_SEC
            and self.buf.span_sec >= WINDOW_SEC * 0.85
        ):
            self._process_epoch(now)

    def _motion_acc(self) -> float:
        """Best estimate of dynamic acceleration for motion gating."""
        candidates = [
            self._sensor_acc_p90,
            self._sensor_acc_rms,
            self._last_acc_rms,
        ]
        return max(c for c in candidates if c > 0) if any(c > 0 for c in candidates) else 0.0

    def _motion_blocks_ppg(self) -> tuple[bool, str]:
        acc = self._motion_acc()
        if acc >= PPG_MOTION_ACC_BLOCK:
            return True, f"acc={acc:.2f}>={PPG_MOTION_ACC_BLOCK}"
        if self._activity_mode in PPG_BLOCKED_MODES:
            return True, f"mode={self._activity_mode}"
        return False, "ok"

    def _publish_pipeline_status(
        self,
        *,
        epoch_skipped: bool,
        skip_reason: str,
        feats: dict | None = None,
    ) -> None:
        now_ms = int(time.time() * 1000)
        if (
            not epoch_skipped
            and skip_reason == "ingest"
            and (time.time() - self._last_pipeline_pub) < 5.0
        ):
            return
        self._last_pipeline_pub = time.time()
        acc = self._motion_acc()
        blocked, _ = self._motion_blocks_ppg()
        payload = {
            "ts": now_ms,
            "buffer_span_sec": round(self.buf.span_sec, 2),
            "buffer_samples": self.buf.sample_count,
            "samples_in_batch": self._last_batch_samples,
            "green_mean": round(self._last_green_mean, 1),
            "batches_total": self._batches_recv,
            "acc_rms": round(acc, 3),
            "activity_mode": self._activity_mode,
            "motion_blocked": blocked,
            "epoch_skipped": epoch_skipped,
            "skip_reason": skip_reason,
        }
        if feats:
            payload.update({
                "rmssd_ppg": feats.get("rmssd"),
                "mean_hr_ppg": feats.get("mean_hr"),
                "peak_count": feats.get("peak_count"),
                "pulse_amp_mean": feats.get("pulse_amp_mean"),
                "z_pulse_amp": feats.get("z_pulse_amp"),
            })
        self.client.publish("biofizic/ppg_pipeline", json.dumps(payload), qos=0)

    def _process_epoch(self, now: float) -> None:
        blocked, block_reason = self._motion_blocks_ppg()
        if blocked:
            self._last_epoch_ts = now
            log.info(
                "PPG Epoch#%d: skip HRV — motion gate (%s)",
                self._epoch_count + 1,
                block_reason,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason=f"motion:{block_reason}")
            return

        self._last_epoch_ts = now
        green, ts_list = self.buf.get_window(WINDOW_SEC)
        green, ts_list, max_gap = trim_after_gaps(green, ts_list)

        if len(green) < FS * 10:
            log.info(
                "PPG Epoch#%d: respins — prea putine samples dupa trim (n=%d, gap=%.1fs)",
                self._epoch_count + 1, len(green), max_gap / 1000.0,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason="few_samples")
            return

        if max_gap > MAX_REAL_GAP_MS:
            log.warning(
                "PPG Epoch#%d: gap real %.1fs — posibil ceas dezlipit",
                self._epoch_count + 1, max_gap / 1000.0,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason="gap")
            return

        span = (ts_list[-1] - ts_list[0]) / 1000.0 if len(ts_list) >= 2 else 0.0
        if span < WINDOW_SEC * 0.8:
            log.info(
                "PPG Epoch#%d: respins — acoperire %.1fs < %.0fs",
                self._epoch_count + 1, span, WINDOW_SEC * 0.8,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason="short_span")
            return

        segments = split_contiguous(ts_list)
        seg_start, seg_end = max(
            segments,
            key=lambda ab: ts_list[ab[1]] - ts_list[ab[0]],
        )
        seg_green = green[seg_start: seg_end + 1]
        seg_ts = ts_list[seg_start: seg_end + 1]
        seg_span = (seg_ts[-1] - seg_ts[0]) / 1000.0 if len(seg_ts) >= 2 else 0.0
        if seg_span < WINDOW_SEC * 0.65:
            log.info(
                "PPG Epoch#%d: respins — segment continuu %.1fs < %.0fs",
                self._epoch_count + 1, seg_span, WINDOW_SEC * 0.65,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason="short_segment")
            return

        uniform = resample_uniform(seg_green, seg_ts, FS)
        if uniform is None or len(uniform) < FS * 10:
            log.info(
                "PPG Epoch#%d: respins — resample esuat (seg=%.1fs n=%s)",
                self._epoch_count + 1, seg_span,
                len(uniform) if uniform is not None else 0,
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason="resample")
            return

        prepped = preprocess_ppg(uniform)
        filtered = apply_filter(prepped)
        feats, reason = extract_hrv_features(filtered, FS)

        if feats is None:
            log.info(
                "PPG Epoch#%d: respins (%s) — span=%.1fs samples=%d",
                self._epoch_count + 1, reason, span, len(green),
            )
            self._publish_pipeline_status(epoch_skipped=True, skip_reason=reason)
            return

        # Logeaza raportul peaks/ibi pentru diagnostic
        peak_efficiency = feats["ibi_n"] / max(feats["peak_count"] - 1, 1)
        if peak_efficiency < 0.6:
            log.warning(
                "PPG Epoch#%d: eficienta IBI scazuta %.0f%% (%d/%d) — posibil double peaks reziduale",
                self._epoch_count + 1,
                peak_efficiency * 100,
                feats["ibi_n"],
                feats["peak_count"] - 1,
            )

        self.baseline.observe(feats["pulse_amp_mean"], self._last_acc_rms)
        z_amp = self.baseline.z_score(feats["pulse_amp_mean"])
        self._epoch_count += 1

        payload = {
            "ts":               int(now * 1000),
            "epoch_n":          self._epoch_count,
            "source":           "ppg_raw",
            "rmssd_ppg":        feats["rmssd"],
            "sdnn_ppg":         feats["sdnn"],
            "pnn50_ppg":        feats["pnn50"],
            "mean_hr_ppg":      feats["mean_hr"],
            "mean_ibi_ppg":     feats["mean_ibi_ms"],
            "ibi_n_ppg":        feats["ibi_n"],
            "peak_count":       feats["peak_count"],
            "pulse_amp_mean":   feats["pulse_amp_mean"],
            "pulse_amp_std":    feats["pulse_amp_std"],
            "z_pulse_amp":      round(z_amp, 3),
            "amp_baseline_ready": self.baseline.ready,
            "ibi_ms_ppg":       feats["ibi_ms"],
            "window_sec":       WINDOW_SEC,
            "buffer_sec":       round(self.buf.total_sec, 1),
        }

        self.client.publish("biofizic/ppg_hrv", json.dumps(payload), qos=1)
        feats["z_pulse_amp"] = round(z_amp, 3)
        self._publish_pipeline_status(epoch_skipped=False, skip_reason="ok", feats=feats)

        log.info(
            "PPG#%d RMSSD=%.1f HR=%.0f amp=%.4f z_amp=%.2f ibi=%d peaks=%d",
            self._epoch_count,
            feats["rmssd"],
            feats["mean_hr"],
            feats["pulse_amp_mean"],
            z_amp,
            feats["ibi_n"],
            feats["peak_count"],
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="PPG raw processor pentru Galaxy Watch 7")
    parser.add_argument("--broker", default="paxbespoke.automateflow.ro")
    parser.add_argument("--port",   type=int, default=1883)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    log.info("Pornire PPG processor (FS=%dHz, window=%ds, buffer=%ds)",
             FS, WINDOW_SEC, BUFFER_SEC)
    PpgEngine(args.broker, args.port).start()


if __name__ == "__main__":
    main()
