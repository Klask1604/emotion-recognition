"""Legacy engines (PPG / WESAD / valence) are enabled for the live research
dashboards. They run in parallel to production and must never raise, even when
fed None, so a missing input degrades gracefully instead of taking down compute."""

from __future__ import annotations

from biofizic.legacy import LegacyEngines, toggles


def test_toggles_enabled_for_research_dashboards():
    # The legacy engines are intentionally on (PPG peaks / WESAD / valence feed
    # the comparison dashboards). This guards against an accidental global off.
    assert toggles.any_enabled()


def test_legacy_engines_report_active():
    # With the research toggles on, the parallel engines report active so the
    # compute service drives them alongside production each tick.
    eng = LegacyEngines()
    assert eng.active
