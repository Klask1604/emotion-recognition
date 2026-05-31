"""Reactivity profile: scales the valence dead-band, persists across reload,
ignores malformed input."""

from __future__ import annotations

from affectus.shared.reactivity import ReactivityProfile


def test_default_is_normal():
    p = ReactivityProfile()
    assert p.level == "normal"
    assert p.deadband_scale == 1.0


def test_deadband_scale_ordering():
    low, normal, high = ReactivityProfile(), ReactivityProfile(), ReactivityProfile()
    low.set("low"); high.set("high")
    # low-responder -> narrower neutral zone; high -> wider.
    assert low.deadband_scale < normal.deadband_scale < high.deadband_scale


def test_unknown_level_ignored():
    p = ReactivityProfile()
    p.set("banana")
    assert p.level == "normal"  # unchanged


def test_persists_across_reload(tmp_path):
    path = tmp_path / "reactivity_profile.json"
    a = ReactivityProfile(path=path)
    a.set("low")
    b = ReactivityProfile(path=path)  # fresh load from disk
    assert b.level == "low"
    assert b.deadband_scale == 0.6
