from __future__ import annotations

from typing import Dict, List, Set, Tuple

from .types import DetectionResult, ProvenanceResult, RiskResult


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _band(score: float) -> str:
    if score >= 0.82:
        return "critical"
    if score >= 0.62:
        return "high"
    if score >= 0.42:
        return "moderate"
    return "low"


def _family_for_signal(signal_name: str, modality: str) -> str:
    s = signal_name.lower()
    if s.startswith("metadata_"):
        return "metadata"
    if s.startswith("quality_"):
        return "quality"
    if s in {"header_consistency", "byte_entropy", "asset_size_mb", "video_stream_count", "audio_stream_count", "video_fps", "video_probe"}:
        return "container"
    if s.startswith("ela_") or s.startswith("patch_") or s.startswith("edge_") or s.startswith("saturation_"):
        return "image_content"
    if s.startswith("audio_"):
        return "audio_content"
    if s.startswith("video_frame"):
        return "video_content"
    if s in {"token_count", "type_token_ratio", "repetition_ratio", "sentence_burstiness", "ai_marker_hits", "zero_width_chars"}:
        return "text_style"
    return modality


def _evidence_summary(detection: DetectionResult, modality: str) -> Tuple[int, int, int, Set[str]]:
    high_count = 0
    medium_count = 0
    suspicious_families: Set[str] = set()

    for sig in detection.signals:
        name = sig.name.lower()
        if name.startswith("quality_"):
            continue
        if sig.score >= 0.7:
            high_count += 1
            suspicious_families.add(_family_for_signal(name, modality))
        elif sig.score >= 0.45:
            medium_count += 1
            suspicious_families.add(_family_for_signal(name, modality))

    return high_count, medium_count, len(suspicious_families), suspicious_families


def _policy_thresholds(profile: str) -> Dict[str, float]:
    if profile == "high_recall":
        return {
            "block": 0.76,
            "priority_review": 0.56,
            "review": 0.36,
            "min_consensus_block": 1,
            "min_quality": 0.30,
            "min_coverage": 0.40,
        }
    if profile == "balanced":
        return {
            "block": 0.82,
            "priority_review": 0.62,
            "review": 0.42,
            "min_consensus_block": 2,
            "min_quality": 0.38,
            "min_coverage": 0.50,
        }
    # industry low false-positive profile
    return {
        "block": 0.86,
        "priority_review": 0.66,
        "review": 0.46,
        "min_consensus_block": 2,
        "min_quality": 0.45,
        "min_coverage": 0.55,
    }


def build_risk(
    detection: DetectionResult,
    provenance: ProvenanceResult,
    modality: str = "binary",
    profile: str = "industry_low_fp",
    threshold_override: float | None = None,
) -> RiskResult:
    thresholds = _policy_thresholds(profile)

    high_count, medium_count, consensus, families = _evidence_summary(detection, modality)

    detection_mix = (
        0.36 * detection.manipulation_likelihood
        + 0.34 * detection.synthetic_likelihood
        + 0.18 * detection.impersonation_likelihood
        + 0.12 * detection.anomaly_likelihood
    )

    provenance_risk = 0.08 if provenance.verified else 0.72
    observability_penalty = _clamp(1.0 - (0.55 * detection.coverage + 0.45 * detection.quality))

    overall = _clamp(
        0.62 * detection_mix
        + 0.30 * provenance_risk
        + 0.08 * observability_penalty
    )

    rationale: List[str] = [
        f"profile={profile}",
        f"detection_mix={detection_mix:.3f}",
        f"detection_coverage={detection.coverage:.3f}",
        f"detection_quality={detection.quality:.3f}",
        f"provenance_verified={provenance.verified}",
        f"provenance_risk={provenance_risk:.3f}",
        f"evidence_high={high_count}",
        f"evidence_medium={medium_count}",
        f"evidence_consensus={consensus}",
        f"evidence_families={','.join(sorted(families)) if families else 'none'}",
    ]

    # False-positive guard: do not escalate to high without multi-family support.
    if consensus < 2 and high_count < 2:
        overall = min(overall, thresholds["priority_review"] - 0.01)
        rationale.append("guardrail: capped score due to insufficient cross-family consensus")

    # Strong provenance with weak artifacts should reduce risk.
    if provenance.verified and high_count <= 1 and detection.synthetic_likelihood < 0.45 and detection.manipulation_likelihood < 0.45:
        overall = _clamp(overall * 0.72)
        rationale.append("guardrail: reduced score due to verified provenance and weak artifact evidence")

    # Escalate only when multiple independent indicators align.
    if (
        not provenance.verified
        and consensus >= 2
        and high_count >= 3
        and detection.coverage >= 0.62
        and detection.quality >= 0.55
    ):
        overall = _clamp(overall + 0.08)
        rationale.append("escalation: multi-family high evidence with sufficient observability")

    if threshold_override is not None:
        calibrated_review = _clamp(float(threshold_override), 0.2, 0.9)
        thresholds["review"] = calibrated_review
        thresholds["priority_review"] = _clamp(calibrated_review + 0.12, 0.25, 0.95)
        thresholds["block"] = _clamp(calibrated_review + 0.24, 0.35, 0.98)
        rationale.append(f"calibrated_review_threshold={calibrated_review:.3f}")

    low_observability = detection.coverage < thresholds["min_coverage"] or detection.quality < thresholds["min_quality"]
    if low_observability:
        overall = min(overall, 0.64)
        rationale.append("guardrail: low observability; verdict constrained to review/inconclusive")

    band = _band(overall)

    if low_observability:
        decision = "inconclusive_review"
    elif overall >= thresholds["block"] and consensus >= int(thresholds["min_consensus_block"]) and high_count >= 3:
        decision = "block_high_risk"
    elif overall >= thresholds["priority_review"] and consensus >= 2:
        decision = "manual_review_priority"
    elif overall >= thresholds["review"]:
        decision = "manual_review"
    else:
        decision = "allow_with_monitoring"

    evidence_count = len(detection.signals) + len(provenance.notes)
    confidence = _clamp(0.22 + 0.30 * detection.coverage + 0.24 * detection.quality + 0.03 * evidence_count)
    if low_observability:
        confidence = _clamp(confidence * 0.82)

    uncertainty = _clamp(1.0 - confidence + 0.15 * observability_penalty)

    return RiskResult(
        overall_risk=overall,
        band=band,
        confidence=confidence,
        uncertainty=uncertainty,
        decision=decision,
        components={
            "detection_mix": detection_mix,
            "provenance_risk": provenance_risk,
            "observability_penalty": observability_penalty,
            "high_signal_count": float(high_count),
            "medium_signal_count": float(medium_count),
            "consensus_count": float(consensus),
        },
        evidence_consensus=consensus,
        rationale=rationale,
    )
