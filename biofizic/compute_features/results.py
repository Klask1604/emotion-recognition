"""Computed HRV metrics and pipeline result types."""

from __future__ import annotations

from dataclasses import dataclass


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
    # Fraction of input beats rejected by the physiological / outlier filter
    # over this window. Standard HRV signal-quality index (Task Force 1996).
    artifact_rate: float = 0.0

    @property
    def is_valid(self) -> bool:
        from biofizic.config import MIN_BEATS_FOR_HRV, MIN_COVERED_SECONDS_FOR_HRV

        return (
            self.beat_count >= MIN_BEATS_FOR_HRV
            and self.covered_seconds >= MIN_COVERED_SECONDS_FOR_HRV
        )


@dataclass(frozen=True)
class MultiWindowHrvResult:
    """Parallel HRV metrics for 30/60/90 second windows."""

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
    # Signal-quality gate outputs (replaced the HAR activity class + confidence).
    motion_state: str          # "still" | "moving" (UI only, not a decision input)
    signal_quality: float      # Q in [0, 1], the decision confidence
    artifact_rate: float
    motion_energy: float
    alert: bool                # CUSUM sustained-stress alert
    decision_reason: str
    baseline_ready: bool
    # Kalman smoother introspection (V.5): the filtered z that arousal is mapped
    # from, and the gain (how much this epoch was trusted). Low gain = the
    # estimate barely moved because quality was poor.
    stress_index_z_filtered: float = 0.0
    kalman_gain: float = 0.0
    # Motion-tolerant fusion: HR-based personal z and the weight given to the
    # HRV channel (w=1 still/clean, ->0 in motion where HR carries the verdict).
    hr_z_score: float = 0.0
    hrv_weight: float = 1.0
    # Multi-channel decision confidence (V.6): the honest confidence of the
    # verdict, blending the HRV channel quality with the HR channel confidence.
    # In motion, signal_quality (HRV-only) collapses but decision_confidence stays
    # high because HR carries the verdict. dominant_channel says which one does:
    # "hrv" (still, precise), "hr" (motion, robust), "blend", or "none" (no HR).
    decision_confidence: float = 0.0
    dominant_channel: str = "hrv"
    # Verdict fidelity. "preliminary" means arousal_10 came from the Kubios
    # population zones (no personal calibration yet); "calibrated" means it
    # came from the personal-z CDF. UIs can gate display badges on this.
    decision_fidelity: str = "calibrated"
    multi_window: MultiWindowHrvResult | None = None


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
    best_window_label: str  # always "w30" (decision); w60/w90 are diagnostic only
    decision: PhysiologyDecision | None
    ibi_buffer_size: int
    motion_state: str
    signal_quality: float
    artifact_rate: float
    baseline_ready: bool
