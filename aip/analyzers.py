"""Text analysis utilities for the AI Text Humanizer.

This module provides heuristic text-feature extraction used by the
evaluation pipeline and the readiness doctor command. All media-forensic
logic (image, audio, video) has been removed; the platform now processes
text exclusively.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from .types import DetectionResult, Signal

try:
    import nltk

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _NLTK_AVAILABLE = False

_TEXT_AI_MARKERS = (
    "as an ai language model",
    "i cannot browse",
    "i do not have real-time",
    "i'm unable to",
    "i can not provide legal advice",
)


def runtime_capabilities() -> Dict[str, bool]:
    """Return a dict showing which NLP dependencies are available at runtime."""
    wordnet_available = False
    punkt_available = False

    if _NLTK_AVAILABLE:
        try:
            nltk.data.find("corpora/wordnet.zip")
            wordnet_available = True
        except LookupError:
            try:
                nltk.data.find("corpora/wordnet")
                wordnet_available = True
            except LookupError:
                pass
        try:
            nltk.data.find("tokenizers/punkt_tab")
            punkt_available = True
        except LookupError:
            pass

    return {
        "nltk": _NLTK_AVAILABLE,
        "wordnet": wordnet_available,
        "punkt": punkt_available,
    }


def sha256_file(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def analyze_text(text: str, identity_claim: Optional[str] = None) -> DetectionResult:
    """Extract heuristic text features and return a :class:`DetectionResult`.

    Signals computed:
    - Type-Token Ratio (TTR) – low TTR indicates repetitive AI prose.
    - Sentence burstiness – low variance indicates uniformly sized AI sentences.
    - Bigram repetition ratio – high ratio indicates templated phrasing.
    - AI marker hits – presence of known AI-generated transition phrases.
    - Zero-width character count – indicates hidden steganographic characters.
    """
    cleaned = text.strip().lower()
    words = re.findall(r"[a-zA-Z0-9']+", cleaned)
    sentences = [s.strip() for s in re.split(r"[.!?]+", cleaned) if s.strip()]

    token_count = len(words)
    unique_count = len(set(words)) if words else 1
    ttr = unique_count / max(1, token_count)

    sentence_lengths = [len(re.findall(r"[a-zA-Z0-9']+", s)) for s in sentences]
    mean_sent = sum(sentence_lengths) / max(1, len(sentence_lengths))
    var_sent = 0.0
    if sentence_lengths:
        var_sent = sum((x - mean_sent) ** 2 for x in sentence_lengths) / len(sentence_lengths)

    burstiness = var_sent / max(1.0, mean_sent)

    bigrams = [tuple(words[i : i + 2]) for i in range(max(0, token_count - 1))]
    repeated_bigrams = sum(1 for _, c in Counter(bigrams).items() if c > 1)
    repetition_ratio = repeated_bigrams / max(1, len(bigrams))

    ai_marker_hits = sum(1 for marker in _TEXT_AI_MARKERS if marker in cleaned)
    punctuation_density = len(re.findall(r"[,:;()\[\]-]", text)) / max(1, len(text))
    zero_width_hits = len(re.findall(r"[\u200B-\u200D\uFEFF]", text))

    synthetic = _clamp(
        0.3 * (1 - _clamp(ttr))
        + 0.25 * _clamp(repetition_ratio * 4)
        + 0.2 * _clamp((0.2 - burstiness) / 0.2)
        + 0.15 * min(1.0, ai_marker_hits / 2)
    )

    manipulation = _clamp(
        0.25 * _clamp(repetition_ratio * 4)
        + 0.2 * (1.0 if punctuation_density > 0.12 else 0.0)
        + 0.25 * _clamp(zero_width_hits / 2)
    )

    impersonation = 0.05
    if identity_claim:
        lc_claim = identity_claim.lower()
        claim_mentions = 1.0 if lc_claim in cleaned else 0.0
        first_person = 1.0 if re.search(r"\bi\b|\bmy\b|\bme\b", cleaned) else 0.0
        impersonation = _clamp(0.1 + 0.45 * claim_mentions * first_person + 0.15 * synthetic)

    anomaly = _clamp(
        0.2 * _clamp(repetition_ratio * 4)
        + 0.25 * _clamp(zero_width_hits / 2)
        + 0.15 * (1.0 if token_count < 3 else 0.0)
    )

    signals: List[Signal] = [
        Signal("token_count", _clamp(token_count / 800), f"token_count={token_count}"),
        Signal("type_token_ratio", 1 - _clamp(ttr), f"ttr={ttr:.3f}"),
        Signal("repetition_ratio", _clamp(repetition_ratio * 4), f"repetition_ratio={repetition_ratio:.3f}"),
        Signal("sentence_burstiness", _clamp((0.8 - burstiness) / 0.8), f"burstiness={burstiness:.3f}"),
        Signal("ai_marker_hits", _clamp(ai_marker_hits / 3), f"ai_marker_hits={ai_marker_hits}"),
        Signal("zero_width_chars", _clamp(zero_width_hits / 2), f"zero_width_hits={zero_width_hits}"),
    ]
    length_quality = _clamp(token_count / 80.0)
    structure_quality = _clamp(len(sentences) / 8.0)
    lexical_quality = _clamp(ttr / 0.5)
    text_quality = _clamp(0.5 * length_quality + 0.3 * structure_quality + 0.2 * lexical_quality)
    signals.append(
        Signal(
            "quality_text_observability",
            text_quality,
            f"tokens={token_count}, sentences={len(sentences)}, ttr={ttr:.3f}",
        )
    )

    return DetectionResult(
        manipulation_likelihood=manipulation,
        synthetic_likelihood=synthetic,
        impersonation_likelihood=impersonation,
        anomaly_likelihood=anomaly,
        coverage=0.95,
        quality=text_quality,
        signals=signals,
    )
