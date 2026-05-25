#!/usr/bin/env python3
"""Generate deterministic ML evaluation report from eval_results JSON files."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "eval_results"
DOCS = ROOT / "docs"


def _load(name: str) -> dict:
    path = EVAL / name
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _ascii_confusion(matrix: list[list[int]], labels: list[str]) -> str:
    lines = ["```", "Predicted ->"]
    header = "Actual \\ Pred | " + " | ".join(f"{l:>10}" for l in labels)
    lines.append(header)
    lines.append("-" * len(header))
    for i, row in enumerate(matrix):
        cells = " | ".join(f"{v:>10}" for v in row)
        lines.append(f"{labels[i]:>12} | {cells}")
    lines.append("```")
    return "\n".join(lines)


def _classification_table(report: dict) -> str:
    lines = ["| Class | Precision | Recall | F1 | Support |", "|-------|-----------|--------|-----|---------|"]
    for label, stats in report.items():
        if not isinstance(stats, dict) or "precision" not in stats:
            continue
        lines.append(
            f"| {label} | {stats['precision']:.3f} | {stats['recall']:.3f} | "
            f"{stats['f1-score']:.3f} | {int(stats['support'])} |"
        )
    return "\n".join(lines)


def main() -> None:
    wisdm = _load("wisdm_har_report.json")
    wesad = _load("wesad_loso_report.json")
    delta = _load("wesad_delta_report.json")

    matrices = {}

    sections = [
        "# ML Evaluation Report (deterministic)",
        "",
        "Production uses **Kubios/Baevsky physiology** (explainable). ML models below are gated off unless LOSO passes thresholds.",
        "",
    ]

    if wisdm:
        gate = wisdm.get("gate", {})
        passed = gate.get("passed", False)
        loso = wisdm.get("loso", {})
        cr = loso.get("classification_report", {})
        sections += [
            "## WISDM HAR (motion)",
            "",
            f"- Gate: **{'PASS' if passed else 'FAIL'}**",
            f"- Accuracy: {gate.get('accuracy', 0):.3f} (min {gate.get('accuracy_min', 0.85)})",
            f"- F1 WALK: {gate.get('f1_walk', 0):.3f}",
            "",
            _classification_table(cr),
            "",
        ]
        matrices["wisdm_har"] = {
            "classes": wisdm.get("loso", {}).get("classes", []),
            "gate_passed": passed,
            "classification_report": cr,
        }

    if wesad:
        gate = wesad.get("gate", {})
        passed = gate.get("passed", False)
        cm = wesad.get("loso", {}).get("confusion_matrix")
        sections += [
            "## WESAD epoch emotion (v3 fusion)",
            "",
            f"- Gate: **{'PASS' if passed else 'FAIL'}**",
            f"- Accuracy: {gate.get('accuracy', 0):.3f}",
            f"- F1 stress: {gate.get('f1_stress', 0):.3f}",
            "",
        ]
        if cm:
            sections.append(_ascii_confusion(cm, ["non_stress", "stress"]))
            sections.append("")
            matrices["wesad_epoch"] = {"confusion_matrix": cm, "gate_passed": passed}

    if delta:
        gate = delta.get("gate", {})
        passed = gate.get("passed", False)
        cr = gate.get("classification_report") or delta.get("loso", {}).get("classification_report", {})
        sections += [
            "## WESAD delta emotion (Solid optional)",
            "",
            f"- Gate: **{'PASS' if passed else 'FAIL'}**",
            f"- Accuracy: {gate.get('accuracy', 0):.3f}",
            f"- F1 stress: {gate.get('f1_stress', 0):.3f}",
            "",
            _classification_table(cr),
            "",
        ]
        matrices["wesad_delta"] = {"gate_passed": passed, "classification_report": cr}

    sections += [
        "## Decision",
        "",
        "All gates failed on consumer GW7 transfer. Deploy path: deterministic HRV + personal baseline + HAR motion cap.",
        "",
    ]

    DOCS.mkdir(parents=True, exist_ok=True)
    report_path = DOCS / "ml_evaluation_report.md"
    report_path.write_text("\n".join(sections), encoding="utf-8")
    json_path = DOCS / "ml_confusion_matrices.json"
    json_path.write_text(json.dumps(matrices, indent=2), encoding="utf-8")
    print(f"Wrote {report_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
