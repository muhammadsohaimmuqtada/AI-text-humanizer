from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .types import ProvenanceResult


def verify_provenance(path: Optional[Path]) -> ProvenanceResult:
    if path is None:
        return ProvenanceResult(
            verified=False,
            confidence=0.2,
            issuer=None,
            generated_by=None,
            chain_of_custody=[],
            notes=["No asset path provided; provenance unavailable"],
        )

    prov_path = Path(str(path) + ".prov.json")
    c2pa_marker = Path(str(path) + ".c2pa")

    if not prov_path.exists() and not c2pa_marker.exists():
        return ProvenanceResult(
            verified=False,
            confidence=0.35,
            issuer=None,
            generated_by=None,
            chain_of_custody=[],
            notes=["No provenance sidecar found", "No C2PA marker found"],
        )

    issuer = None
    generated_by = None
    chain = []
    signature_valid = False
    notes = []

    if prov_path.exists():
        try:
            payload = json.loads(prov_path.read_text(encoding="utf-8"))
            signature_valid = bool(payload.get("signature_valid", False))
            issuer = payload.get("issuer")
            generated_by = payload.get("generated_by")
            chain = payload.get("chain_of_custody") or []
            if not isinstance(chain, list):
                chain = []
                notes.append("chain_of_custody present but invalid format")
        except (OSError, json.JSONDecodeError):
            notes.append("Failed to parse provenance sidecar")

    if c2pa_marker.exists():
        notes.append("C2PA marker sidecar present")

    verified = signature_valid or c2pa_marker.exists()
    confidence = 0.85 if verified else 0.45

    if verified and not issuer:
        notes.append("Verified marker found but issuer missing")

    return ProvenanceResult(
        verified=verified,
        confidence=confidence,
        issuer=issuer,
        generated_by=generated_by,
        chain_of_custody=chain,
        notes=notes,
    )
