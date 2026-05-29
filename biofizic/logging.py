"""Human-readable decision logs for Docker / debugging.

Layout per epoch:
  [SUMMARY]  one-line verdict — the key factors, no formulas
  [BASELINE] one decisional parameter per line
  [QUALITY]  signal-quality / gating factors
  [WINDOW]   multi-window RMSSD
A blank line separates consecutive blocks.
"""

from __future__ import annotations

from biofizic.compute_features.results import MultiWindowHrvResult, PhysiologyDecision
from biofizic.config import ARTIFACT_RATE_MAX


def format_summary_line(decision: PhysiologyDecision) -> str:
    """Epoch verdict in plain factors (no formulas)."""
    alert = " ALERT" if decision.alert else ""
    return (
        f"[SUMMARY] arousal {decision.display_arousal_10}/10 {decision.display_label}{alert}"
        f" | hr {decision.mean_heart_rate_bpm:.0f} bpm"
        f" | rmssd {decision.rmssd_ms:.0f} ms"
        f" | stress index {decision.kubios_stress_index:.1f} ({decision.kubios_label})"
        f" | motion {decision.motion_state}"
        f" | confidence {decision.decision_confidence * 100:.0f}% (via {decision.dominant_channel})"
    )


def format_baseline_block(decision: PhysiologyDecision) -> list[str]:
    """One decisional parameter per line."""
    return [
        f"[BASELINE] ready              = {'yes' if decision.baseline_ready else 'no'}",
        f"[BASELINE] personal_stress_SI = {decision.baseline_stress_index:.1f}",
        f"[BASELINE] z_raw HRV           = {decision.stress_index_z_score:+.2f}",
        f"[BASELINE] z_raw HR            = {decision.hr_z_score:+.2f}",
        f"[BASELINE] hrv_weight (fusion) = {decision.hrv_weight:.2f}  (1=still/HRV, 0=motion/HR)",
        f"[BASELINE] z_filtered (Kalman) = {decision.stress_index_z_filtered:+.2f}",
        f"[BASELINE] kalman_gain         = {decision.kalman_gain:.2f}  (epoch trust; low => held)",
        f"[BASELINE] personal_label      = {decision.baseline_label}",
        f"[BASELINE] matches_kubios      = {'yes' if decision.labels_agree else 'no'}",
    ]


def format_quality_block(decision: PhysiologyDecision) -> list[str]:
    return [
        f"[QUALITY] motion        = {decision.motion_state}",
        f"[QUALITY] motion_energy = {decision.motion_energy:.3f}",
        f"[QUALITY] artifact_rate = {decision.artifact_rate * 100:.0f}%"
        f"  (epoch usable if <={ARTIFACT_RATE_MAX * 100:.0f}% and still)",
        f"[QUALITY] hrv_quality Q  = {decision.signal_quality:.2f}  (HRV-only; collapses in motion)",
        f"[QUALITY] confidence     = {decision.decision_confidence:.2f}  (multi-channel; via {decision.dominant_channel})",
        f"[QUALITY] alert (CUSUM) = {'YES' if decision.alert else 'no'}",
        f"[QUALITY] reason        = {decision.decision_reason}",
    ]


def format_window_line(multi: MultiWindowHrvResult | None) -> str:
    if multi is None:
        return "[WINDOW] no_data"
    parts = []
    for name, metrics in (
        ("w30", multi.window_30_seconds),
        ("w60", multi.window_60_seconds),
        ("w90", multi.window_90_seconds),
    ):
        if metrics and metrics.is_valid:
            parts.append(f"{name} rmssd={metrics.rmssd_ms:.0f}")
        else:
            parts.append(f"{name} --")
    return "[WINDOW] " + " | ".join(parts) + "   (w60 decides; w30/w90 diagnostic)"


def format_ibi_breakdown(multi: MultiWindowHrvResult | None) -> str:
    """Per-window IBI counts + artifact rates so the user can see exactly
    where beats are getting rejected. Beats received  → kept after the
    artifact filter; the ratio is the published artifact_rate."""
    if multi is None:
        return "[IBI] no_data"
    parts = []
    for name, metrics in (
        ("w30", multi.window_30_seconds),
        ("w60", multi.window_60_seconds),
        ("w90", multi.window_90_seconds),
    ):
        if metrics:
            kept = metrics.beat_count
            rejected_pct = metrics.artifact_rate * 100
            # received ≈ kept / (1 - artifact_rate); avoid div-by-zero.
            received = (
                kept / (1.0 - metrics.artifact_rate)
                if metrics.artifact_rate < 0.999 else kept
            )
            parts.append(f"{name} in={received:.0f} kept={kept} art={rejected_pct:.0f}%")
        else:
            parts.append(f"{name} --")
    return "[IBI] " + " | ".join(parts)


def format_decision_block(decision: PhysiologyDecision) -> str:
    lines = [
        format_summary_line(decision),
        *format_baseline_block(decision),
        *format_quality_block(decision),
        format_window_line(decision.multi_window),
        format_ibi_breakdown(decision.multi_window),
    ]
    # Trailing newline => one blank line between consecutive blocks.
    return "\n".join(lines) + "\n"
