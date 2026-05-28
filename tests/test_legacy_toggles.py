"""Legacy engines (PPG / WESAD / valence) are off by default — they are
chest-strap-trained (WESAD) or ad-hoc (valence) and produce biased output on
wrist-only PPG. They run in parallel to production ONLY when explicitly
toggled on for a specific research session. They must never raise even when
fed None, so a missing input degrades gracefully instead of taking down
compute."""

from __future__ import annotations

from biofizic.legacy import LegacyEngines, toggles


def test_toggles_off_by_default():
    # The production default is off. Documented in toggles.py.
    assert not toggles.any_enabled()


def test_legacy_engines_inactive_when_toggles_off():
    # With all toggles off, the parallel engines report inactive so the
    # compute service skips them.
    eng = LegacyEngines()
    assert not eng.active


def test_legacy_engines_become_active_when_toggle_flipped(monkeypatch):
    # Flipping any toggle on must activate the engine surface so a research
    # session can be opted in without code changes elsewhere.
    monkeypatch.setattr(toggles, "ENABLE_VALENCE", True)
    monkeypatch.setattr(toggles, "ENABLE_RAW_PPG", True)
    eng = LegacyEngines()
    assert eng.active
