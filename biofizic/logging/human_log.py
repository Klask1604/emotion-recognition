"""Human-readable log lines for Docker and debugging."""

from __future__ import annotations

from biofizic.types.samples import MultiWindowHrvResult, PhysiologyDecision


def format_physiology_line(decision: PhysiologyDecision) -> str:
    return (
        f"[PHYSIO] activation_level={decision.display_arousal_10}/10 "
        f"({decision.display_label}) | "
        f"heart_rate={decision.mean_heart_rate_bpm:.0f}bpm | "
        f"rmssd={decision.rmssd_ms:.1f}ms | "
        f"stress_index={decision.kubios_stress_index:.1f} ({decision.kubios_label})"
    )


def format_baseline_line(decision: PhysiologyDecision) -> str:
    base = decision.baseline_stress_index
    agree = "yes" if decision.labels_agree else "no"
    return (
        f"[BASELINE] ready={'yes' if decision.baseline_ready else 'no'} | "
        f"personal_stress_index={base:.1f} | "
        f"z_score={decision.stress_index_z_score:+.2f} | "
        f"labels_match={agree} (kubios={decision.kubios_label} "
        f"baseline={decision.baseline_label})"
    )


def format_motion_line(decision: PhysiologyDecision) -> str:
    return (
        f"[MOTION] activity={decision.motion_class} "
        f"conf={decision.motion_confidence:.2f} | "
        f"context={decision.activity_mode} | "
        f"reason={decision.decision_reason}"
    )


def format_window_line(multi: MultiWindowHrvResult | None) -> str:
    if multi is None:
        return "[WINDOW] no_data"
    parts = []
    for name, metrics in (
        ("w15", multi.window_15_seconds),
        ("w30", multi.window_30_seconds),
        ("w60", multi.window_60_seconds),
        ("w90", multi.window_90_seconds),
    ):
        if metrics and metrics.is_valid:
            parts.append(f"{name}:rmssd={metrics.rmssd_ms:.0f}")
        else:
            parts.append(f"{name}:--")
    return "[WINDOW] " + " ".join(parts)


def format_affect_line(decision: PhysiologyDecision) -> str:
    return (
        f"[AFFECT] valence={decision.valence_10}/10 "
        f"({decision.valence_label}) | "
        f"quadrant={decision.affect_quadrant} | "
        f"z_pulse_amp={decision.z_pulse_amp:+.2f}"
    )


def format_decision_block(decision: PhysiologyDecision) -> str:
    lines = [
        format_physiology_line(decision),
        format_motion_line(decision),
        format_baseline_line(decision),
        format_affect_line(decision),
        format_window_line(decision.multi_window),
    ]
    return "\n".join(lines)
