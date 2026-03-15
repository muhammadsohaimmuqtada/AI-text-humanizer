from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from .analyzers import analyze_file, analyze_text, guess_modality, sha256_file
from .calibration import resolve_threshold_override
from .provenance import verify_provenance
from .risk import build_risk
from .types import AnalysisResult, AssetInfo


def analyze(
    input_path: Optional[str] = None,
    text: Optional[str] = None,
    modality: str = "auto",
    identity_claim: Optional[str] = None,
    policy_profile: str = "industry_low_fp",
    calibration: Optional[dict[str, Any]] = None,
) -> AnalysisResult:
    path = Path(input_path).expanduser().resolve() if input_path else None

    if path and not path.exists():
        raise FileNotFoundError(f"Input path not found: {path}")

    actual_modality = guess_modality(path, modality)

    if text is not None:
        detection = analyze_text(text, identity_claim)
    elif path is not None:
        if actual_modality == "text":
            raw_text = path.read_text(encoding="utf-8", errors="replace")
            detection = analyze_text(raw_text, identity_claim)
        else:
            detection = analyze_file(path, actual_modality, identity_claim)
    else:
        raise ValueError("Either --input or --text must be provided")

    provenance = verify_provenance(path)
    threshold_override = resolve_threshold_override(
        calibration=calibration,
        profile=policy_profile,
        modality=actual_modality,
    )

    risk = build_risk(
        detection,
        provenance,
        modality=actual_modality,
        profile=policy_profile,
        threshold_override=threshold_override,
    )

    asset = AssetInfo(
        path=str(path) if path else None,
        modality=actual_modality,
        sha256=sha256_file(path) if path else None,
        size_bytes=path.stat().st_size if path else None,
        extension=path.suffix.lower() if path else None,
    )

    forensics = []
    for s in detection.signals:
        forensics.append(f"{s.name}: score={s.score:.3f}, detail={s.detail}")
    forensics.append(f"detection_coverage: {detection.coverage:.3f}")
    forensics.append(f"detection_quality: {detection.quality:.3f}")
    for n in provenance.notes:
        forensics.append(f"provenance_note: {n}")
    forensics.append(f"decision: {risk.decision}")
    forensics.append(f"uncertainty: {risk.uncertainty:.3f}")
    if threshold_override is not None:
        forensics.append(f"calibration_threshold_override: {threshold_override:.3f}")
    forensics.extend(risk.rationale)

    return AnalysisResult(
        asset=asset,
        detection=detection,
        provenance=provenance,
        risk=risk,
        forensics=forensics,
    )
