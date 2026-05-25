"""Structured samples and multi-window pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class InterbeatIntervalEntry:
    interval_ms: int
    timestamp_ms: int | None = None


@dataclass
class IbiBatchMessage:
    """One second of IBI data from the watch."""

    timestamp_ms: int
    intervals_ms: list[int] = field(default_factory=list)
    timestamps_ms: list[int] = field(default_factory=list)


@dataclass
class PpgBatchMessage:
    """One second of PPG samples from the watch."""

    timestamp_ms: int
    green: list[int] = field(default_factory=list)
    infrared: list[int] = field(default_factory=list)
    sample_timestamps_ms: list[int] = field(default_factory=list)


@dataclass
class SensorBatchMessage:
    """Aggregated sensor stats from the watch (1 Hz)."""

    timestamp_ms: int
    heart_rate_bpm: float = 0.0
    acceleration_rms: float = 0.0
    acceleration_p90: float = 0.0
    acceleration_std: float = 0.0
    gyroscope_rms: float = 0.0
    gyroscope_p90: float = 0.0
    gyroscope_std: float = 0.0
    skin_temperature_c: float = 0.0
    ambient_temperature_c: float = 0.0
    display_on: bool = True


@dataclass(frozen=True)
class HrvMetrics:
    """Heart rate variability metrics for one time window."""

    rmssd_ms: float
    sdnn_ms: float
    mean_interbeat_interval_ms: float
    mean_heart_rate_bpm: float
    pnn50_percent: float
    beat_count: int
    covered_seconds: float
    baevsky_stress_index_raw: float = 0.0
    kubios_stress_index: float = 0.0

    @property
    def is_valid(self) -> bool:
        from biofizic.config import MIN_BEATS_FOR_HRV, MIN_COVERED_SECONDS_FOR_HRV

        return (
            self.beat_count >= MIN_BEATS_FOR_HRV
            and self.covered_seconds >= MIN_COVERED_SECONDS_FOR_HRV
        )


@dataclass(frozen=True)
class MultiWindowHrvResult:
    """Parallel HRV metrics for 15/30/60/90 second windows."""

    window_15_seconds: HrvMetrics | None
    window_30_seconds: HrvMetrics | None
    window_60_seconds: HrvMetrics | None
    window_90_seconds: HrvMetrics | None


@dataclass(frozen=True)
class PhysiologyDecision:
    """Final physiology output for one decision tick."""

    display_label: str
    display_arousal_10: int
    kubios_label: str
    baseline_label: str
    labels_agree: bool
    kubios_stress_index: float
    baseline_stress_index: float
    stress_index_z_score: float
    rmssd_ms: float
    mean_heart_rate_bpm: float
    motion_class: str
    motion_confidence: float
    activity_mode: str
    decision_reason: str
    baseline_ready: bool
    multi_window: MultiWindowHrvResult | None = None
    valence_10: int = 5
    valence_label: str = "neutral"
    affect_quadrant: str = "calm"
    z_pulse_amp: float = 0.0


@dataclass
class WindowResult:
    rmssd_ms: float
    sdnn_ms: float
    pnn50_pct: float
    kubios_stress_index: float
    mean_hr_bpm: float
    quality: str  # "full" | "partial" | "unavailable"
    ibi_count: int
    covered_seconds: float

    @staticmethod
    def unavailable() -> WindowResult:
        return WindowResult(
            rmssd_ms=0.0,
            sdnn_ms=0.0,
            pnn50_pct=0.0,
            kubios_stress_index=0.0,
            mean_hr_bpm=0.0,
            quality="unavailable",
            ibi_count=0,
            covered_seconds=0.0,
        )


@dataclass
class MultiWindowResult:
    ts: float
    w30: WindowResult
    w60: WindowResult
    w90: WindowResult
    best: WindowResult
    best_window_label: str  # "w30" | "w60" | "w90"
    decision: PhysiologyDecision | None
    ibi_buffer_size: int
    motion_class: str
    baseline_ready: bool
