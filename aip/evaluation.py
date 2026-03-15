from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .engine import analyze


_POSITIVE_LABELS = {
    "1",
    "true",
    "yes",
    "fake",
    "manipulated",
    "synthetic",
    "deepfake",
    "impostor",
}
_NEGATIVE_LABELS = {
    "0",
    "false",
    "no",
    "real",
    "authentic",
    "genuine",
    "clean",
}


@dataclass
class EvalSample:
    sample_id: str
    modality: str
    label: int
    score: float
    decision: str
    coverage: float
    quality: float
    provenance_verified: bool


@dataclass
class Metrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    fpr: float
    fnr: float
    accuracy: float
    f1: float


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def _parse_label(raw: str) -> int:
    value = (raw or "").strip().lower()
    if value in _POSITIVE_LABELS:
        return 1
    if value in _NEGATIVE_LABELS:
        return 0
    raise ValueError(f"Unsupported label value: {raw!r}")


def _compute_metrics(labels: List[int], preds: List[int]) -> Metrics:
    tp = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 1)
    fp = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 1)
    tn = sum(1 for y, p in zip(labels, preds) if y == 0 and p == 0)
    fn = sum(1 for y, p in zip(labels, preds) if y == 1 and p == 0)

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    specificity = _safe_div(tn, tn + fp)
    fpr = _safe_div(fp, fp + tn)
    fnr = _safe_div(fn, fn + tp)
    accuracy = _safe_div(tp + tn, len(labels))
    f1 = _safe_div(2 * precision * recall, precision + recall)

    return Metrics(
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        specificity=specificity,
        fpr=fpr,
        fnr=fnr,
        accuracy=accuracy,
        f1=f1,
    )


def _as_dict(metrics: Metrics) -> Dict[str, float]:
    return {
        "tp": metrics.tp,
        "fp": metrics.fp,
        "tn": metrics.tn,
        "fn": metrics.fn,
        "precision": round(metrics.precision, 6),
        "recall": round(metrics.recall, 6),
        "specificity": round(metrics.specificity, 6),
        "fpr": round(metrics.fpr, 6),
        "fnr": round(metrics.fnr, 6),
        "accuracy": round(metrics.accuracy, 6),
        "f1": round(metrics.f1, 6),
    }


def _iter_rows(dataset_csv: Path) -> Iterable[Dict[str, str]]:
    with dataset_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            clean: Dict[str, str] = {}
            for k, v in row.items():
                if k is None:
                    continue
                if isinstance(v, list):
                    clean[k] = ",".join(str(x) for x in v).strip()
                else:
                    clean[k] = str(v or "").strip()
            yield clean


def run_dataset(
    dataset_csv: str,
    profile: str = "industry_low_fp",
) -> Dict[str, object]:
    path = Path(dataset_csv).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    required = {"label"}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        headers = set(reader.fieldnames or [])
    missing = sorted(required - headers)
    if missing:
        raise ValueError(f"Dataset missing required columns: {', '.join(missing)}")

    samples: List[EvalSample] = []
    skipped: List[Dict[str, str]] = []

    for i, row in enumerate(_iter_rows(path), start=1):
        sample_id = row.get("id") or f"row-{i}"
        input_path = row.get("input_path") or row.get("path") or ""
        text = row.get("text") or ""
        modality = row.get("modality") or "auto"
        identity_claim = row.get("identity_claim") or None

        if not input_path and not text:
            skipped.append({"id": sample_id, "reason": "missing input_path/text"})
            continue

        try:
            label = _parse_label(row.get("label", ""))
        except ValueError as exc:
            skipped.append({"id": sample_id, "reason": str(exc)})
            continue

        try:
            result = analyze(
                input_path=input_path or None,
                text=text or None,
                modality=modality,
                identity_claim=identity_claim,
                policy_profile=profile,
            )
        except Exception as exc:  # keep evaluation robust for batch runs
            skipped.append({"id": sample_id, "reason": f"analyze_failed: {exc}"})
            continue

        samples.append(
            EvalSample(
                sample_id=sample_id,
                modality=result.asset.modality,
                label=label,
                score=result.risk.overall_risk,
                decision=result.risk.decision,
                coverage=result.detection.coverage,
                quality=result.detection.quality,
                provenance_verified=result.provenance.verified,
            )
        )

    return {
        "dataset_path": str(path),
        "profile": profile,
        "samples": [s.__dict__ for s in samples],
        "skipped": skipped,
    }


def _threshold_grid(scores: List[float]) -> List[float]:
    coarse = [i / 100 for i in range(20, 96, 2)]
    around_scores: List[float] = []
    for s in scores:
        around_scores.extend([_clamp(s - 0.03), _clamp(s), _clamp(s + 0.03)])
    values = sorted(set(round(v, 4) for v in (coarse + around_scores)))
    return [v for v in values if 0.0 <= v <= 1.0]


def _modality_metrics(samples: List[EvalSample], threshold: float) -> Dict[str, Dict[str, float]]:
    by_modality: Dict[str, List[EvalSample]] = {}
    for s in samples:
        by_modality.setdefault(s.modality, []).append(s)

    out: Dict[str, Dict[str, float]] = {}
    for modality, group in sorted(by_modality.items()):
        labels = [x.label for x in group]
        preds = [1 if x.score >= threshold else 0 for x in group]
        out[modality] = _as_dict(_compute_metrics(labels, preds))
        out[modality]["count"] = len(group)
    return out


def _recommended_threshold_for_samples(samples: List[EvalSample], target_fpr: float) -> Tuple[float, Metrics, str]:
    labels = [s.label for s in samples]
    scores = [s.score for s in samples]
    sweep_rows = []
    for t in _threshold_grid(scores):
        preds = [1 if s >= t else 0 for s in scores]
        m = _compute_metrics(labels, preds)
        sweep_rows.append((t, m))

    meeting_target = [(t, m) for t, m in sweep_rows if m.fpr <= target_fpr]
    if meeting_target:
        t, m = max(meeting_target, key=lambda x: (x[1].recall, x[1].f1, x[1].precision))
        return t, m, "max_recall_under_fpr_target"
    t, m = min(sweep_rows, key=lambda x: (x[1].fpr, -x[1].recall))
    return t, m, "no_threshold_met_target_fpr; chose_lowest_fpr"


def evaluate_thresholds(
    raw_results: Dict[str, object],
    threshold: float,
    target_fpr: float = 0.03,
) -> Dict[str, object]:
    sample_dicts = raw_results.get("samples", [])
    samples: List[EvalSample] = [EvalSample(**d) for d in sample_dicts]

    if not samples:
        return {
            "error": "No evaluable samples",
            "dataset_path": raw_results.get("dataset_path"),
            "profile": raw_results.get("profile"),
            "skipped": raw_results.get("skipped", []),
        }

    labels = [s.label for s in samples]
    scores = [s.score for s in samples]
    preds_at_input = [1 if s.score >= threshold else 0 for s in samples]
    metrics_at_input = _compute_metrics(labels, preds_at_input)

    sweep_rows = []
    for t in _threshold_grid(scores):
        preds = [1 if s >= t else 0 for s in scores]
        m = _compute_metrics(labels, preds)
        sweep_rows.append({
            "threshold": t,
            "precision": m.precision,
            "recall": m.recall,
            "fpr": m.fpr,
            "f1": m.f1,
            "accuracy": m.accuracy,
        })

    rec_t, rec_metrics, selection_reason = _recommended_threshold_for_samples(samples, target_fpr=target_fpr)

    avg_coverage = _safe_div(sum(s.coverage for s in samples), len(samples))
    avg_quality = _safe_div(sum(s.quality for s in samples), len(samples))
    inconclusive_rate = _safe_div(sum(1 for s in samples if s.decision == "inconclusive_review"), len(samples))
    provenance_verified_rate = _safe_div(sum(1 for s in samples if s.provenance_verified), len(samples))

    modality_breakdown = _modality_metrics(samples, rec_t)
    per_modality_thresholds: Dict[str, Dict[str, object]] = {}
    by_modality: Dict[str, List[EvalSample]] = {}
    for s in samples:
        by_modality.setdefault(s.modality, []).append(s)
    for modality, group in sorted(by_modality.items()):
        t_m, m_m, reason_m = _recommended_threshold_for_samples(group, target_fpr=target_fpr)
        per_modality_thresholds[modality] = {
            "threshold": round(t_m, 6),
            "reason": reason_m,
            "metrics": _as_dict(m_m),
            "count": len(group),
        }

    top_sweep = sorted(sweep_rows, key=lambda x: (x["fpr"], -x["recall"], -x["precision"]))[:10]

    return {
        "dataset_path": raw_results.get("dataset_path"),
        "profile": raw_results.get("profile"),
        "sample_count": len(samples),
        "skipped_count": len(raw_results.get("skipped", [])),
        "skipped": raw_results.get("skipped", []),
        "input_threshold": threshold,
        "metrics_at_input_threshold": _as_dict(metrics_at_input),
        "recommended_threshold": {
            "value": rec_t,
            "reason": selection_reason,
            "target_fpr": target_fpr,
            "metrics": _as_dict(rec_metrics),
        },
        "top_threshold_candidates": [
            {
                "threshold": round(x["threshold"], 4),
                "precision": round(x["precision"], 6),
                "recall": round(x["recall"], 6),
                "fpr": round(x["fpr"], 6),
                "f1": round(x["f1"], 6),
                "accuracy": round(x["accuracy"], 6),
            }
            for x in top_sweep
        ],
        "quality_summary": {
            "avg_coverage": round(avg_coverage, 6),
            "avg_quality": round(avg_quality, 6),
            "inconclusive_rate": round(inconclusive_rate, 6),
            "provenance_verified_rate": round(provenance_verified_rate, 6),
        },
        "modality_breakdown": modality_breakdown,
        "per_modality_thresholds": per_modality_thresholds,
        "samples": sample_dicts,
    }


def write_json_report(payload: Dict[str, object], output_path: str) -> None:
    path = Path(output_path).expanduser().resolve()
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_calibration_bundle(report: Dict[str, object]) -> Dict[str, object]:
    profile = str(report.get("profile", "industry_low_fp"))
    default_threshold = float(
        (report.get("recommended_threshold", {}) or {}).get("value", 0.65)
    )
    per_modality = report.get("per_modality_thresholds", {}) or {}
    modalities: Dict[str, Dict[str, float]] = {}
    if isinstance(per_modality, dict):
        for modality, info in per_modality.items():
            if not isinstance(info, dict):
                continue
            try:
                t = float(info.get("threshold", default_threshold))
            except Exception:
                t = default_threshold
            modalities[str(modality)] = {"threshold": round(_clamp(t), 6)}

    return {
        "version": "1",
        "source_dataset": report.get("dataset_path"),
        "generated_from_profile": profile,
        "default_threshold": round(_clamp(default_threshold), 6),
        "profiles": {
            profile: {
                "default_threshold": round(_clamp(default_threshold), 6),
                "modalities": modalities,
            }
        },
    }
