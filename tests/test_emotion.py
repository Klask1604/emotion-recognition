"""The emotion verdict: Russell quadrants + neutral dead-band, codes matching the
Grafana circumplex dashboard (0 Neutru, 1 Calm, 2 Trist, 3 Bucuros, 4 Stresat)."""

from __future__ import annotations

from affectus.shared.emotion import emotion_verdict, ValenceVerdictStabilizer

DB = 0.2  # a representative dead-band


def test_happy_high_arousal_positive():
    assert emotion_verdict(0.8, 0.5, DB, True) == ("Bucuros", 3)


def test_stressed_high_arousal_negative():
    assert emotion_verdict(0.8, -0.5, DB, True) == ("Stresat", 4)


def test_calm_low_arousal_positive():
    assert emotion_verdict(-0.8, 0.5, DB, True) == ("Calm", 1)


def test_sad_low_arousal_negative():
    assert emotion_verdict(-0.8, -0.5, DB, True) == ("Trist", 2)


def test_neutral_inside_deadband():
    # |valence| < deadband -> neutral regardless of arousal.
    assert emotion_verdict(1.0, 0.1, DB, True) == ("Neutru", 0)
    assert emotion_verdict(-1.0, -0.15, DB, True) == ("Neutru", 0)


def test_neutral_when_valence_not_ready():
    # Before the valence baseline locks, never assert a quadrant.
    assert emotion_verdict(0.8, 0.9, DB, False) == ("Neutru", 0)


def test_boundary_arousal_zero_is_high_side():
    # z exactly 0 counts as the activated side (>=).
    assert emotion_verdict(0.0, 0.5, DB, True) == ("Bucuros", 3)


def test_boundary_valence_at_deadband_edge():
    # |v| == deadband is NOT inside the neutral zone (strict <), so it's a verdict.
    assert emotion_verdict(0.5, DB, DB, True) == ("Bucuros", 3)


# ── ValenceVerdictStabilizer ─────────────────────────────────────────────────

def test_low_confidence_epochs_stay_neutral():
    # Even a strongly-positive valence stays Neutru when the model is unsure
    # (confidence below the gate) — the uncertain epoch is pulled to 0.
    s = ValenceVerdictStabilizer()
    last = None
    for _ in range(6):
        last = s.update(arousal_z=0.8, valence_personal=0.9, confidence=0.5,
                        deadband=DB, valence_ready=True)
    assert last == ("Neutru", 0)


def test_single_noisy_epoch_does_not_flip_verdict():
    # Settle on Bucuros, then one negative spike should NOT flip to Stresat
    # (hysteresis needs 3 consecutive). Confidence is high so it's not gated.
    s = ValenceVerdictStabilizer()
    for _ in range(5):
        s.update(0.8, 0.8, 0.9, DB, True)            # settle Bucuros
    assert s.update(0.8, 0.8, 0.9, DB, True) == ("Bucuros", 3)
    out = s.update(0.8, -0.8, 0.9, DB, True)         # one negative blip
    assert out == ("Bucuros", 3)                      # held by median + hysteresis


def test_sustained_change_flips_after_hysteresis():
    s = ValenceVerdictStabilizer()
    for _ in range(6):
        s.update(0.8, 0.8, 0.9, DB, True)            # settle Bucuros (high arousal +)
    # Sustained NEGATIVE (still high arousal) -> should eventually flip to Stresat.
    outs = [s.update(0.8, -0.8, 0.9, DB, True) for _ in range(8)]
    assert outs[-1] == ("Stresat", 4)


def test_reset_clears_state():
    s = ValenceVerdictStabilizer()
    for _ in range(5):
        s.update(0.8, 0.8, 0.9, DB, True)
    s.reset()
    # After reset the first epoch is adopted immediately (no prior displayed).
    assert s.update(-0.8, -0.8, 0.9, DB, True) == ("Trist", 2)
