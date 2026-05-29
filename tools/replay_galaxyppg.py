#!/usr/bin/env python3
"""
GalaxyPPG dataset replay through the production pipeline.

For each subject:
  - Read GalaxyWatch/HR.csv  (IBI batches per ~4 s burst)
  - Read GalaxyWatch/ACC.csv (25 Hz acceleration)
  - Read PolarH10/IBI.csv    (ground-truth ECG-derived IBI)
  - Read Event.csv           (session labels per timestamp)

Feeds the IBI batches into PhysiologyPipeline exactly like the production
compute_engine receives them from the watch, and records the arousal_10 /
RMSSD / signal_quality trajectory. Returns also the Polar H10 ground-truth
RMSSD per 30 s window for direct comparison.

No MQTT, no Docker — in-process replay over PhysiologyPipeline so we can
batch the 24 subjects offline. The pipeline state is reset between subjects.
"""

from __future__ import annotations

import ast
import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import biofizic._bootstrap  # noqa: F401

from biofizic.engine.baseline import RestBaselineStore
from biofizic.engine.pipeline import PhysiologyPipeline
from biofizic.ingestion.messages import (
    AcquisitionBatchMessage,
    InterbeatIntervalEntry,
)

DATASET_ROOT = ROOT / "datasets" / "galaxyppg" / "Dataset"

# Replay cadence: pipeline expects 1 Hz acquisition/batch (matches production).
TICK_INTERVAL_MS = 1000

# Map each session to an expected arousal regime. NaN for "no specific
# expectation" (analysed but not scored). Values are integers 1..10 matching
# the pipeline's display_arousal_10.
SESSION_EXPECTED_AROUSAL: dict[str, float] = {
    "adaptation": float("nan"),
    "baseline": 3.0,         # rest reference
    "tsst-prep": 5.0,
    "tsst-speech": 8.0,      # acute social stress
    "meditation-1": 2.0,
    "screen-reading": 4.0,
    "ssst-prep": 5.0,
    "ssst-sing": 7.0,        # social stress test (sing)
    "meditation-2": 2.0,
    "keyboard-typing": 4.0,
    "rest-1": 3.0,
    "mobile-typing": 4.0,
    "rest-2": 3.0,
    "standing": 4.0,
    "rest-3": 3.0,
    "walking": 6.0,
    "rest-4": 3.0,
    "jogging": 8.0,
    "rest-5": 4.0,
    "running": 9.0,
}


@dataclass
class PolarIbi:
    ts_ms: int
    interval_ms: int


@dataclass
class HrBurst:
    """One row from GalaxyWatch/HR.csv — anchor timestamp + list of IBIs."""
    anchor_ts_ms: int
    intervals_ms: list[int]
    statuses: list[int]


@dataclass
class AccSample:
    ts_ms: int
    magnitude: float


@dataclass
class SessionEvent:
    ts_ms: int
    session: str
    status: str  # "ENTER" / "EXIT"


# ── Parsers ──────────────────────────────────────────────────────────────────

def _parse_int_list(s: str) -> list[int]:
    if not s or s == "[]":
        return []
    try:
        v = ast.literal_eval(s)
    except (SyntaxError, ValueError):
        return []
    return [int(x) for x in v]


def load_hr_bursts(path: Path) -> list[HrBurst]:
    bursts: list[HrBurst] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            ts_str = row.get("timestamp")
            if not ts_str:
                continue
            try:
                ts = int(ts_str)
            except ValueError:
                continue
            intervals = _parse_int_list(row.get("ibi") or "")
            statuses = _parse_int_list(row.get("ibiStatus") or "")
            if not intervals:
                continue
            if len(statuses) < len(intervals):
                statuses = statuses + [0] * (len(intervals) - len(statuses))
            bursts.append(HrBurst(anchor_ts_ms=ts, intervals_ms=intervals, statuses=statuses))
    return bursts


def load_polar_ibi(path: Path) -> list[PolarIbi]:
    out: list[PolarIbi] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["phoneTimestamp"])
                dur = int(row["duration"])
            except (KeyError, ValueError):
                continue
            out.append(PolarIbi(ts_ms=ts, interval_ms=dur))
    return out


def load_acc(path: Path) -> list[AccSample]:
    out: list[AccSample] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["timestamp"])
                x = float(row["x"])
                y = float(row["y"])
                z = float(row["z"])
            except (KeyError, ValueError):
                continue
            # Dynamic acceleration magnitude minus 1 g (m/s² units).
            mag = abs(math.sqrt(x * x + y * y + z * z) - 9.81)
            out.append(AccSample(ts_ms=ts, magnitude=mag))
    return out


def load_events(path: Path) -> list[SessionEvent]:
    out: list[SessionEvent] = []
    with path.open() as f:
        for row in csv.DictReader(f):
            try:
                ts = int(row["timestamp"])
            except (KeyError, ValueError):
                continue
            out.append(SessionEvent(ts_ms=ts, session=row["session"], status=row["status"]))
    return out


# ── Helpers ──────────────────────────────────────────────────────────────────

def burst_to_entries(burst: HrBurst) -> list[InterbeatIntervalEntry]:
    """Walk-back the IBI list from the burst anchor, mirroring the watch's
    AcquisitionAssembler.buildIbiTimestamps logic. Drops entries with
    status != 0 (Samsung-reported errors)."""
    end_ts = burst.anchor_ts_ms
    out: list[InterbeatIntervalEntry] = []
    # Walk backwards so the latest beat is at anchor_ts.
    pairs = list(zip(burst.intervals_ms, burst.statuses))
    for interval, status in reversed(pairs):
        if status in (0, -1):  # -1 also accepted per IbiSignalFilter
            out.append(InterbeatIntervalEntry(interval_ms=interval, timestamp_ms=end_ts))
        end_ts -= interval
    return list(reversed(out))


def session_for_ts(events: list[SessionEvent], ts_ms: int) -> str:
    """Find the active session at ts_ms by scanning ENTER/EXIT pairs."""
    active = ""
    for ev in events:
        if ev.ts_ms > ts_ms:
            break
        if ev.status == "ENTER":
            active = ev.session
        elif ev.status == "EXIT" and ev.session == active:
            active = ""
    return active


def rmssd_from_intervals(intervals: list[int]) -> float:
    if len(intervals) < 2:
        return 0.0
    a = np.asarray(intervals, dtype=float)
    diffs = np.diff(a)
    return float(np.sqrt(np.mean(diffs * diffs)))


def polar_rmssd_at(polar: list[PolarIbi], end_ts_ms: int, window_ms: int = 30_000) -> float:
    start = end_ts_ms - window_ms
    intervals = [p.interval_ms for p in polar if start <= p.ts_ms <= end_ts_ms]
    return rmssd_from_intervals(intervals)


# ── Replay ───────────────────────────────────────────────────────────────────

@dataclass
class SubjectResult:
    subject: str
    samples: list[dict] = field(default_factory=list)
    skipped_reason: str | None = None


class _InMemoryBaseline(RestBaselineStore):
    """In-memory baseline so replay across subjects doesn't touch
    data/rest_baseline.json on disk. Subclassing keeps the API the same as
    production while making _load / _save no-ops."""

    def __init__(self) -> None:
        from collections import deque as _deque

        from biofizic.config import BASELINE_ROBUST_WINDOW_EPOCHS

        self._path = Path("/dev/null")
        self._ln_rmssd: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_si: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self._ln_hr: deque[float] = _deque(maxlen=BASELINE_ROBUST_WINDOW_EPOCHS)
        self.is_ready = False
        self.rest_observation_count = 0
        self.reported_baseline_arousal = 0.5

    def _load(self) -> None:  # noqa: D401  pragma: no cover
        pass

    def _save(self) -> None:
        pass


def _make_pipeline() -> PhysiologyPipeline:
    p = PhysiologyPipeline()
    p.baseline = _InMemoryBaseline()
    return p


def _acc_window_stats(samples: list[AccSample], end_ts_ms: int, window_ms: int = 1000) -> tuple[float, float, float]:
    """Quick replay-side acc stats: (rms, p90, std) over last `window_ms`."""
    start = end_ts_ms - window_ms
    vals = [s.magnitude for s in samples if start <= s.ts_ms <= end_ts_ms]
    if not vals:
        return 0.0, 0.0, 0.0
    arr = np.asarray(vals, dtype=float)
    rms = float(np.sqrt(np.mean(arr * arr)))
    p90 = float(np.percentile(arr, 90))
    std = float(np.std(arr))
    return rms, p90, std


def replay_subject(subject_dir: Path) -> SubjectResult:
    """Stream one subject's data through the production pipeline and return
    the per-tick arousal_10 / RMSSD trajectory paired with the active session
    label and the Polar H10 ground-truth RMSSD at each tick."""
    subj_id = subject_dir.name
    result = SubjectResult(subject=subj_id)

    hr_csv = subject_dir / "GalaxyWatch" / "HR.csv"
    polar_csv = subject_dir / "PolarH10" / "IBI.csv"
    acc_csv = subject_dir / "GalaxyWatch" / "ACC.csv"
    events_csv = subject_dir / "Event.csv"
    if not hr_csv.exists() or not polar_csv.exists() or not events_csv.exists():
        result.skipped_reason = "missing required CSVs (P01 has no GalaxyWatch data)"
        return result

    bursts = load_hr_bursts(hr_csv)
    polar = load_polar_ibi(polar_csv)
    events = load_events(events_csv)
    acc = load_acc(acc_csv) if acc_csv.exists() else []
    if not bursts or not polar or not events:
        result.skipped_reason = "empty primary CSV"
        return result

    bursts.sort(key=lambda b: b.anchor_ts_ms)
    polar.sort(key=lambda p: p.ts_ms)
    events.sort(key=lambda e: e.ts_ms)
    acc.sort(key=lambda a: a.ts_ms)

    # PolarH10 logger stores phoneTimestamp in KST treated as Unix epoch ms
    # (UTC+9 offset baked into the values), while GalaxyWatch logger writes
    # real UTC ms. Detect the rounded-hour offset and bring Polar onto the
    # GalaxyWatch clock so RMSSD windows align. Only apply if the offset is
    # a whole-hour multiple; anything else is treated as small clock drift.
    raw_offset_ms = polar[0].ts_ms - bursts[0].anchor_ts_ms
    hour_ms = 3600_000
    offset_hours = round(raw_offset_ms / hour_ms)
    if offset_hours != 0:
        shift_ms = offset_hours * hour_ms
        polar = [PolarIbi(ts_ms=p.ts_ms - shift_ms, interval_ms=p.interval_ms) for p in polar]

    pipeline = _make_pipeline()

    # Clip replay to the session range (first event ENTER → last event EXIT).
    # Before this we either had no IBIs (waste of CPU) or the Polar tail kept
    # ticking ~9 h after GW ended due to the offset, producing 600 min of
    # garbage. Sessions are the only time window where comparison is valid.
    t0 = events[0].ts_ms
    t_end = events[-1].ts_ms
    next_burst_idx = 0
    seq = 0
    tick_ts = t0

    while tick_ts <= t_end:
        # Drain HR bursts whose anchor falls in the last second — emulates the
        # 1 Hz acquisition/batch publish cadence on the watch.
        ibi_entries: list[InterbeatIntervalEntry] = []
        while next_burst_idx < len(bursts) and bursts[next_burst_idx].anchor_ts_ms <= tick_ts:
            ibi_entries.extend(burst_to_entries(bursts[next_burst_idx]))
            next_burst_idx += 1

        # Motion summary from the GalaxyWatch ACC stream around this tick.
        acc_rms, acc_p90, acc_std = _acc_window_stats(acc, tick_ts)

        synth = AcquisitionBatchMessage(
            timestamp_publish_ms=tick_ts,
            timestamp_anchor_ms=tick_ts,
            sequence=seq,
            heart_rate_bpm=0.0,  # GalaxyWatch HR field is unreliable here; pipeline
                                 # uses 0 → HR channel inactive → HRV channel only.
            display_on=True,
            acceleration_rms=acc_rms,
            acceleration_p90=acc_p90,
            acceleration_std=acc_std,
            # Without an FFT on raw ACC we approximate cardiac-band energy by
            # acc_rms (motion_energy falls back to this in the pipeline). Good
            # enough for still/moving classification — the goal is HRV validity
            # comparison, not motion taxonomy.
            acc_band_cardiac=acc_rms * 0.1,
            ibi_intervals_ms=[e.interval_ms for e in ibi_entries],
            ibi_timestamps_ms=[int(e.timestamp_ms or 0) for e in ibi_entries],
            ibi_timestamp_source="dataset_replay",
        )
        seq += 1
        pipeline.ingest_acquisition(synth)
        run_result = pipeline.run(
            now=tick_ts / 1000.0, end_timestamp_ms=tick_ts, publish_epoch=True,
        )

        decision = run_result.decision
        polar_rmssd = polar_rmssd_at(polar, tick_ts)
        result.samples.append({
            "ts_ms": tick_ts,
            "session": session_for_ts(events, tick_ts),
            "arousal_10": int(decision.display_arousal_10) if decision else None,
            "our_rmssd": float(decision.rmssd_ms) if decision else None,
            "polar_rmssd": polar_rmssd,
            "our_mean_hr": float(decision.mean_heart_rate_bpm) if decision else None,
            "signal_quality": float(decision.signal_quality) if decision else 0.0,
            "decision_fidelity": decision.decision_fidelity if decision else "none",
            "baseline_ready": pipeline.baseline.is_ready,
            "ibi_buffer_size": run_result.ibi_buffer_size,
        })

        tick_ts += TICK_INTERVAL_MS

    return result


def list_subjects() -> list[Path]:
    if not DATASET_ROOT.is_dir():
        return []
    return sorted(p for p in DATASET_ROOT.iterdir() if p.is_dir() and p.name.startswith("P"))


def main() -> None:
    import argparse, json
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", help="One subject (e.g. P02). Default: all.")
    parser.add_argument("--out", default="eval_results/galaxyppg_replay.jsonl",
                        help="JSONL output path under repo root.")
    args = parser.parse_args()

    subjects = (
        [DATASET_ROOT / args.subject] if args.subject else list_subjects()
    )
    out_path = ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        for sd in subjects:
            print(f"[replay] {sd.name}", flush=True)
            res = replay_subject(sd)
            if res.skipped_reason:
                print(f"  skipped: {res.skipped_reason}")
                continue
            for row in res.samples:
                row["subject"] = res.subject
                fh.write(json.dumps(row) + "\n")
            print(f"  {len(res.samples)} samples written")
    print(f"\nDone. Output: {out_path}")


if __name__ == "__main__":
    main()
