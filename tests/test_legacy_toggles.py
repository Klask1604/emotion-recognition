"""Legacy engines (PPG / WESAD / valence) are off by default — they are
chest-strap-trained (WESAD) or ad-hoc (valence) and produce biased output on
wrist-only PPG. They run in parallel to production ONLY when explicitly
toggled on for a specific research session. They must never raise even when
fed None, so a missing input degrades gracefully instead of taking down
compute."""

from __future__ import annotations

from biofizic.legacy import LegacyEngines, toggles


def test_legacy_output_never_enters_production_decision():
    # Whatever the toggles, legacy/research output is surfaced only on
    # biofizic/legacy/* via LegacyOutputs — it must never appear as a field on
    # the production PhysiologyDecision that drives VR.
    from dataclasses import fields
    from biofizic.compute_features.results import PhysiologyDecision

    decision_fields = {f.name for f in fields(PhysiologyDecision)}
    for leaked in ("p_stress", "valence", "respiration", "rsa_bpm", "ppg_bpm"):
        assert leaked not in decision_fields


def test_legacy_engines_inactive_when_all_toggles_off(monkeypatch):
    # With every toggle forced off, the parallel engines report inactive so the
    # compute service skips them entirely (the "production light" path).
    for flag in (
        "ENABLE_RAW_PPG", "ENABLE_PPG_PEAKS", "ENABLE_WESAD",
        "ENABLE_VALENCE", "ENABLE_RESPIRATION_COMPARE",
    ):
        monkeypatch.setattr(toggles, flag, False)
    eng = LegacyEngines()
    assert not eng.active


def test_legacy_engines_become_active_when_toggle_flipped(monkeypatch):
    # Flipping any toggle on must activate the engine surface so a research
    # session can be opted in without code changes elsewhere.
    monkeypatch.setattr(toggles, "ENABLE_VALENCE", True)
    monkeypatch.setattr(toggles, "ENABLE_RAW_PPG", True)
    eng = LegacyEngines()
    assert eng.active
