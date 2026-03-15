from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

from .analyzers import runtime_capabilities


def _score_band(score: float) -> str:
    if score >= 85:
        return "Market Candidate"
    if score >= 70:
        return "Pilot Ready"
    if score >= 55:
        return "Alpha"
    return "Research"


def _bool_to_pass(v: bool) -> str:
    return "PASS" if v else "FAIL"


def _metric(report: Dict[str, object], section: str, key: str, default: float = 0.0) -> float:
    obj = report.get(section, {})
    if not isinstance(obj, dict):
        return default
    raw = obj.get(key, default)
    try:
        return float(raw)
    except Exception:
        return default


def build_readiness_scorecard(report: Dict[str, object]) -> Dict[str, object]:
    metrics = report.get("recommended_threshold", {}).get("metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}

    precision = float(metrics.get("precision", 0.0))
    recall = float(metrics.get("recall", 0.0))
    fpr = float(metrics.get("fpr", 1.0))
    accuracy = float(metrics.get("accuracy", 0.0))

    avg_coverage = _metric(report, "quality_summary", "avg_coverage", 0.0)
    avg_quality = _metric(report, "quality_summary", "avg_quality", 0.0)
    inconclusive_rate = _metric(report, "quality_summary", "inconclusive_rate", 1.0)
    provenance_rate = _metric(report, "quality_summary", "provenance_verified_rate", 0.0)

    caps = runtime_capabilities()
    sample_count = int(report.get("sample_count", 0) or 0)

    gate_precision = precision >= 0.90
    gate_recall = recall >= 0.70
    gate_fpr = fpr <= 0.03
    gate_coverage = avg_coverage >= 0.70
    gate_quality = avg_quality >= 0.55
    gate_inconclusive = inconclusive_rate <= 0.25
    gate_dataset_size = sample_count >= 200

    perf_score = min(40.0, 16.0 * precision + 12.0 * recall + 8.0 * (1.0 - fpr) + 4.0 * accuracy)
    obs_score = min(20.0, 10.0 * avg_coverage + 7.0 * avg_quality + 3.0 * (1.0 - inconclusive_rate))
    prov_score = min(10.0, 10.0 * provenance_rate)
    tooling_score = (
        2.5 * float(caps.get("exiftool", False))
        + 2.5 * float(caps.get("ffprobe", False))
        + 2.5 * float(caps.get("ffmpeg", False))
        + 2.5 * float(caps.get("numpy", False) and caps.get("pillow", False) and caps.get("scipy_wav", False))
    )

    ops_score = 0.0
    if gate_dataset_size:
        ops_score += 8.0
    if gate_inconclusive:
        ops_score += 6.0
    if "recommended_threshold" in report:
        ops_score += 6.0

    total = round(perf_score + obs_score + prov_score + tooling_score + ops_score, 2)
    band = _score_band(total)

    gates = {
        "precision>=0.90": gate_precision,
        "recall>=0.70": gate_recall,
        "fpr<=0.03": gate_fpr,
        "avg_coverage>=0.70": gate_coverage,
        "avg_quality>=0.55": gate_quality,
        "inconclusive_rate<=0.25": gate_inconclusive,
        "sample_count>=200": gate_dataset_size,
    }

    blocked_by = [k for k, v in gates.items() if not v]

    return {
        "readiness_score": total,
        "readiness_band": band,
        "component_scores": {
            "performance_40": round(perf_score, 2),
            "observability_20": round(obs_score, 2),
            "provenance_10": round(prov_score, 2),
            "tooling_10": round(tooling_score, 2),
            "ops_20": round(ops_score, 2),
        },
        "gates": gates,
        "blocked_by": blocked_by,
        "capabilities": caps,
        "recommended_next_steps": _recommendations(blocked_by),
    }


def _recommendations(blocked_by: list[str]) -> list[str]:
    recommendations = []
    if "sample_count>=200" in blocked_by:
        recommendations.append("Expand labeled benchmark set to at least 200 samples per critical modality.")
    if "fpr<=0.03" in blocked_by:
        recommendations.append("Retune threshold for lower false positives and validate on out-of-domain negatives.")
    if "precision>=0.90" in blocked_by:
        recommendations.append("Raise decision threshold or tighten consensus rule for high-risk decisions.")
    if "recall>=0.70" in blocked_by:
        recommendations.append("Add stronger model signals to catch subtle manipulations.")
    if "avg_coverage>=0.70" in blocked_by:
        recommendations.append("Install missing local integrations (ffprobe/ffmpeg) and ensure full pipeline execution.")
    if "avg_quality>=0.55" in blocked_by:
        recommendations.append("Enforce minimum media quality requirements before final decisions.")
    if "inconclusive_rate<=0.25" in blocked_by:
        recommendations.append("Improve ingest quality and add fallback detectors to reduce inconclusive outcomes.")
    if not recommendations:
        recommendations.append("Proceed to limited pilot with continuous monitoring and weekly threshold checks.")
    return recommendations


def render_scorecard_markdown(report: Dict[str, object], scorecard: Dict[str, object]) -> str:
    lines = []
    lines.append("# Authenticity Platform Market Readiness Scorecard")
    lines.append("")
    lines.append(f"- Dataset: `{report.get('dataset_path', 'unknown')}`")
    lines.append(f"- Profile: `{report.get('profile', 'unknown')}`")
    lines.append(f"- Samples evaluated: `{report.get('sample_count', 0)}`")
    lines.append(f"- Readiness Score: **{scorecard['readiness_score']} / 100**")
    lines.append(f"- Readiness Band: **{scorecard['readiness_band']}**")
    lines.append("")

    lines.append("## Gate Results")
    for gate, ok in scorecard.get("gates", {}).items():
        lines.append(f"- {_bool_to_pass(bool(ok))}: `{gate}`")
    lines.append("")

    lines.append("## Component Scores")
    comp = scorecard.get("component_scores", {})
    for k, v in comp.items():
        lines.append(f"- `{k}`: {v}")
    lines.append("")

    rec = report.get("recommended_threshold", {})
    metrics = rec.get("metrics", {}) if isinstance(rec, dict) else {}
    lines.append("## Recommended Threshold")
    lines.append(f"- Threshold: `{rec.get('value', 'n/a')}`")
    lines.append(f"- Selection Reason: `{rec.get('reason', 'n/a')}`")
    lines.append(f"- Precision: `{metrics.get('precision', 'n/a')}`")
    lines.append(f"- Recall: `{metrics.get('recall', 'n/a')}`")
    lines.append(f"- FPR: `{metrics.get('fpr', 'n/a')}`")
    lines.append("")

    lines.append("## Next Steps")
    for item in scorecard.get("recommended_next_steps", []):
        lines.append(f"- {item}")
    lines.append("")

    return "\n".join(lines)


def write_scorecard_files(
    report: Dict[str, object],
    scorecard: Dict[str, object],
    json_path: str,
    markdown_path: str,
) -> Tuple[str, str]:
    jp = Path(json_path).expanduser().resolve()
    mp = Path(markdown_path).expanduser().resolve()
    jp.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")
    mp.write_text(render_scorecard_markdown(report, scorecard), encoding="utf-8")
    return str(jp), str(mp)
