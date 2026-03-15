from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Signal:
    name: str
    score: float
    detail: str


@dataclass
class AssetInfo:
    path: Optional[str]
    modality: str
    sha256: Optional[str]
    size_bytes: Optional[int]
    extension: Optional[str]


@dataclass
class DetectionResult:
    manipulation_likelihood: float
    synthetic_likelihood: float
    impersonation_likelihood: float
    anomaly_likelihood: float
    coverage: float = 0.5
    quality: float = 0.5
    signals: List[Signal] = field(default_factory=list)


@dataclass
class ProvenanceResult:
    verified: bool
    confidence: float
    issuer: Optional[str]
    generated_by: Optional[str]
    chain_of_custody: List[str]
    notes: List[str] = field(default_factory=list)


@dataclass
class RiskResult:
    overall_risk: float
    band: str
    confidence: float
    uncertainty: float
    decision: str
    components: Dict[str, float]
    evidence_consensus: int
    rationale: List[str]


@dataclass
class AnalysisResult:
    asset: AssetInfo
    detection: DetectionResult
    provenance: ProvenanceResult
    risk: RiskResult
    forensics: List[str]

    def to_dict(self) -> Dict[str, Any]:
        out = asdict(self)
        out["detection"]["signals"] = [asdict(s) for s in self.detection.signals]
        return out
