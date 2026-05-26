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
