"""
Watch-side flicker fix: streak-based hysteresis on the live arousal integer.

The compute engine publishes biofizic/state/live at 1 Hz but the rolling 30 s
HRV buffer recomputes Kubios SI every second; near a zone boundary the
rounded arousal_10 would alternate 2-3-2-3 even with a stable signal. The
hysteresis only adopts a new integer after LIVE_AROUSAL_HYSTERESIS_TICKS
consecutive ticks have agreed on it. These tests pin that behaviour.
"""

from __future__ import annotations

from biofizic.config import LIVE_AROUSAL_HYSTERESIS_TICKS
from biofizic.compute_features.results import (
    MultiWindowResult,
    PhysiologyDecision,
    WindowResult,
)
from biofizic.engine.pipeline import PhysiologyPipeline
from services.compute_engine import ComputeEngineService


def _fresh_service() -> ComputeEngineService:
    svc = ComputeEngineService.__new__(ComputeEngineService)
    svc._live_displayed_a10 = None
    svc._live_candidate_a10 = None
    svc._live_candidate_streak = 0
    return svc


def test_pure_alternation_does_not_flicker():
    svc = _fresh_service()
    outputs = [svc._update_live_arousal_hysteresis(v) for v in [3, 2, 3, 2, 3, 2, 3]]
    # Once the first value is adopted, the alternation keeps proposing the
    # opposite value but never reaches the streak required to switch.
    assert outputs[0] == 3
    assert all(o == 3 for o in outputs)


def test_sustained_change_flips_after_required_streak():
    svc = _fresh_service()
    # Settle at 3, then push 4 sustained.
    for _ in range(5):
        assert svc._update_live_arousal_hysteresis(3) == 3
    # The first N-1 ticks of the new value should not flip yet.
    for _ in range(LIVE_AROUSAL_HYSTERESIS_TICKS - 1):
        assert svc._update_live_arousal_hysteresis(4) == 3
    # The Nth confirms and flips.
    assert svc._update_live_arousal_hysteresis(4) == 4


def test_two_tick_blip_does_not_flip_at_hysteresis_3():
    """With LIVE_AROUSAL_HYSTERESIS_TICKS=3 a two-tick blip stays hidden."""
    svc = _fresh_service()
    for _ in range(3):
        assert svc._update_live_arousal_hysteresis(5) == 5
    # Two ticks of 7 is not enough to flip if hysteresis is at least 3.
    if LIVE_AROUSAL_HYSTERESIS_TICKS >= 3:
        assert svc._update_live_arousal_hysteresis(7) == 5
        assert svc._update_live_arousal_hysteresis(7) == 5
        assert svc._update_live_arousal_hysteresis(5) == 5


def test_isolated_spike_is_ignored():
    svc = _fresh_service()
    for _ in range(3):
        assert svc._update_live_arousal_hysteresis(5) == 5
    # A one-tick spike to 8 with no follow-up must not break the display.
    assert svc._update_live_arousal_hysteresis(8) == 5
    assert svc._update_live_arousal_hysteresis(5) == 5


# ---------------------------------------------------------------------------
# window_used must reflect the window that ACTUALLY drove the verdict, not a
# hardcoded label. Regression guard for the w30/w60 incoherence: the pipeline
# decides on w60 (w30 only as a cold-start fallback), so the published payload
# must say so, otherwise the windows dashboard misattributes the decision.
# ---------------------------------------------------------------------------

def _service_with_pipeline() -> ComputeEngineService:
    svc = _fresh_service()
    svc.pipeline = PhysiologyPipeline()
    svc._anchor_ms = 1_716_000_000_000
    return svc


def _minimal_decision() -> PhysiologyDecision:
    return PhysiologyDecision(
        display_label="Echilibrat",
        display_arousal_10=5,
        kubios_label="Echilibrat",
        baseline_label="Echilibrat",
        labels_agree=True,
        kubios_stress_index=10.0,
        baseline_stress_index=10.0,
        stress_index_z_score=0.0,
        rmssd_ms=42.0,
        mean_heart_rate_bpm=70.0,
        motion_state="still",
        signal_quality=0.9,
        artifact_rate=0.05,
        motion_energy=0.01,
        alert=False,
        decision_reason="test",
        baseline_ready=True,
    )


def _result_with_label(label: str) -> MultiWindowResult:
    w = WindowResult(
        rmssd_ms=42.0, sdnn_ms=50.0, pnn50_pct=10.0, kubios_stress_index=10.0,
        mean_hr_bpm=70.0, quality="full", ibi_count=60, covered_seconds=60.0,
    )
    return MultiWindowResult(
        ts=0.0, w30=w, w60=w, w90=w, best=w, best_window_label=label,
        decision=None, ibi_buffer_size=60, motion_state="still",
        signal_quality=0.9, artifact_rate=0.05, baseline_ready=True,
    )


def test_window_used_reports_w60_when_decision_runs_on_w60():
    svc = _service_with_pipeline()
    payload = svc._decision_payload(
        _minimal_decision(), live=False, result=_result_with_label("w60")
    )
    assert payload["window_used"] == "w60", "must not hardcode w30 when w60 decides"


def test_window_used_reports_w30_fallback_during_cold_start():
    svc = _service_with_pipeline()
    payload = svc._decision_payload(
        _minimal_decision(), live=True, result=_result_with_label("w30")
    )
    assert payload["window_used"] == "w30"
