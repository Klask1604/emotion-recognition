"""VR scene context -> valence prior, and the server handler that caches it.

Valence in VR is derived from the scene's visual cues (bright/green = pleasant,
dark/red/chaotic = unpleasant), combined server-side with the MEASURED arousal.
The verdict marks valence_source="context" so it never claims to read valence
from the pulse. Fresh context (<10 s) feeds the verdict; stale/absent falls back."""

from __future__ import annotations

import time

from affectus.shared.scene_context import context_to_valence_prior, valence_prior_label
from services.compute_engine import ComputeEngineService


def test_pleasant_scene_positive_prior():
    # bright, green/blue, calm -> pleasant
    p = context_to_valence_prior(light=0.85, r=0.3, g=0.7, b=0.6, motion=0.1)
    assert p > 0.3
    assert valence_prior_label(p) == "pleasant"


def test_threatening_scene_negative_prior():
    # dark, red, chaotic -> unpleasant
    p = context_to_valence_prior(light=0.15, r=0.85, g=0.1, b=0.1, motion=0.8)
    assert p < -0.3
    assert valence_prior_label(p) == "unpleasant"


def test_neutral_scene_near_zero():
    p = context_to_valence_prior(light=0.5, r=0.45, g=0.45, b=0.45, motion=0.2)
    assert abs(p) < 0.3
    assert valence_prior_label(p) == "neutral"


def test_missing_input_is_failsafe_zero():
    assert context_to_valence_prior(light=None, r=0.5, g=0.5, b=0.5, motion=0.5) == 0.0


def test_handler_caches_context_with_prior():
    svc = ComputeEngineService.__new__(ComputeEngineService)
    svc._last_context = None
    now = 1000.0
    svc._handle_context(
        {"scene": "Forest", "light": 0.9, "r": 0.3, "g": 0.7, "b": 0.6, "motion": 0.0},
        now,
    )
    ctx = svc._last_context
    assert ctx is not None
    assert ctx["scene"] == "Forest"
    assert ctx["valence_prior"] > 0.3          # pleasant scene
    assert ctx["label"] == "pleasant"
    assert ctx["ts"] == now


def test_handler_ignores_bad_numbers_gracefully():
    svc = ComputeEngineService.__new__(ComputeEngineService)
    svc._last_context = None
    # non-numeric cues default to neutral, never crash
    svc._handle_context({"scene": "X", "light": "bad", "motion": None}, 5.0)
    assert svc._last_context is not None
    assert svc._last_context["scene"] == "X"
