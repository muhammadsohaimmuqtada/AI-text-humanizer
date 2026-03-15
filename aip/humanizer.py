"""AI Text Humanizer – core NLP transformation pipeline.

The pipeline applies four transformations in order:

1. **Marker stripping** – removes common AI transition phrases
   (e.g. "In conclusion,", "Furthermore,", "As an AI language model").
2. **Sentence tokenisation** – splits the cleaned text into individual sentences.
3. **Burstiness variation** – randomly merges adjacent sentences so that sentence
   lengths vary, a key trait of human writing.
4. **Synonym substitution** – replaces a fraction of adjectives and adverbs with
   WordNet synonyms to raise lexical perplexity without changing meaning.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

try:
    import nltk
    from nltk.corpus import wordnet
    from nltk.tokenize import sent_tokenize, word_tokenize

    _NLTK_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    _NLTK_AVAILABLE = False

# ---------------------------------------------------------------------------
# AI transition marker list (case-insensitive)
# ---------------------------------------------------------------------------

_AI_MARKERS: List[str] = [
    "in conclusion,",
    "in conclusion",
    "it is important to note that",
    "it is worth noting that",
    "it is crucial to understand that",
    "it should be noted that",
    "it is essential to note that",
    "as an ai language model,",
    "as an ai language model",
    "as an ai,",
    "as an ai",
    "i cannot browse",
    "i do not have real-time",
    "additionally,",
    "furthermore,",
    "moreover,",
    "ultimately,",
    "in summary,",
    "to summarize,",
    "in other words,",
    "as previously mentioned,",
    "as previously mentioned",
    "needless to say,",
    "first and foremost,",
    "first and foremost",
    "last but not least,",
    "last but not least",
    "it goes without saying that",
    "to put it simply,",
    "to put it simply",
]

# POS tags eligible for synonym substitution (adjectives and adverbs only)
_SWAP_TAGS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

# Conjunctions used when merging two adjacent short sentences
_MERGE_CONJUNCTIONS = [
    " and ",
    " while ",
    ", meaning ",
    " — ",
    ", which ",
]


# ---------------------------------------------------------------------------
# Public result type
# ---------------------------------------------------------------------------


@dataclass
class HumanizeResult:
    """Result of a text humanization pass."""

    original_text: str
    humanized_text: str
    original_word_count: int
    humanized_word_count: int
    markers_removed: int
    sentences_merged: int


# ---------------------------------------------------------------------------
# NLTK helper
# ---------------------------------------------------------------------------


def _ensure_nltk_data() -> bool:
    """Download required NLTK corpora if not already present.

    Returns ``True`` when all required data is available, ``False`` otherwise.
    """
    if not _NLTK_AVAILABLE:
        return False

    needed = [
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
        ("corpora/wordnet.zip", "wordnet"),
    ]
    for resource_path, download_name in needed:
        try:
            nltk.data.find(resource_path)
        except LookupError:
            try:
                nltk.download(download_name, quiet=True)
            except Exception:  # pragma: no cover - network may be unavailable
                return False
    return True


# ---------------------------------------------------------------------------
# Step 1 – AI marker stripping
# ---------------------------------------------------------------------------


def strip_ai_markers(text: str) -> Tuple[str, int]:
    """Remove common AI transition phrases from *text*.

    Markers are matched case-insensitively and removed from longest to
    shortest to prevent partial substring conflicts.

    Returns:
        ``(cleaned_text, count)`` where *count* is the total number of
        marker occurrences that were removed.
    """
    count = 0
    result = text

    for marker in sorted(_AI_MARKERS, key=len, reverse=True):
        pattern = re.compile(re.escape(marker), re.IGNORECASE)
        result, n = pattern.subn("", result)
        count += n

    # Collapse double-spaces introduced by removals
    result = re.sub(r"  +", " ", result).strip()
    return result, count


# ---------------------------------------------------------------------------
# Step 2 – Sentence tokenisation
# ---------------------------------------------------------------------------


def _tokenize_sentences(text: str) -> List[str]:
    """Split *text* into sentences using NLTK ``sent_tokenize`` when available.

    Falls back to a simple regex split on ``.``, ``!``, and ``?`` if NLTK is
    unavailable or its tokenizer data has not been downloaded.
    """
    if _NLTK_AVAILABLE and _ensure_nltk_data():
        try:
            return sent_tokenize(text)
        except Exception:
            pass

    # Simple fallback: split on sentence-ending punctuation
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# Step 3 – Burstiness variation (sentence merging)
# ---------------------------------------------------------------------------


def vary_sentence_lengths(
    sentences: List[str],
    merge_rate: float = 0.25,
    rng: Optional[random.Random] = None,
) -> Tuple[List[str], int]:
    """Randomly merge pairs of adjacent sentences to increase burstiness.

    AI-generated text tends to use sentences of similar lengths. Varying the
    lengths by occasionally combining short consecutive sentences with a
    conjunction makes the text read more naturally.

    Args:
        sentences: List of individual sentences.
        merge_rate: Probability (0–1) that a sentence is merged with its
            immediate successor.
        rng: Optional seeded :class:`random.Random` for reproducible output.

    Returns:
        A ``(modified_list, merge_count)`` tuple where *merge_count* is the
        total number of merges performed.
    """
    _rng = rng or random
    result: List[str] = []
    merges = 0
    i = 0

    while i < len(sentences):
        if i < len(sentences) - 1 and _rng.random() < merge_rate:
            current = sentences[i]
            next_sent = sentences[i + 1]
            conjunction = _rng.choice(_MERGE_CONJUNCTIONS)
            # Strip trailing punctuation from the first sentence
            base = re.sub(r"[.!?]+$", "", current.rstrip())
            # Lower-case the first letter of the next sentence after conjunction
            tail = next_sent.lstrip()
            if len(tail) > 1:
                tail = tail[0].lower() + tail[1:]
            elif tail:
                tail = tail[0].lower()
            result.append(base + conjunction + tail)
            merges += 1
            i += 2
        else:
            result.append(sentences[i])
            i += 1

    return result, merges


# ---------------------------------------------------------------------------
# Step 4 – Synonym substitution (perplexity raising)
# ---------------------------------------------------------------------------


def _get_synonym(word: str, rng: Optional[random.Random] = None) -> Optional[str]:
    """Return a single WordNet synonym for *word*, or ``None`` if unavailable.

    Only adjectives (JJ*) and adverbs (RB*) are passed in from the caller, so
    the synonym lookup is safe with respect to preserving logical meaning.
    """
    if not _NLTK_AVAILABLE:
        return None

    synonyms: set = set()
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            name = lemma.name()
            if name.lower() != word.lower() and "_" not in name and name.isalpha():
                synonyms.add(name)

    if not synonyms:
        return None

    # Prefer synonyms that don't share the same 3-character stem as the
    # original word, to maximise vocabulary diversity.
    stem_prefix = word.lower()[:min(len(word), 3)]
    candidates = [s for s in synonyms if not s.lower().startswith(stem_prefix)]
    pool = candidates if candidates else list(synonyms)
    _rng = rng or random
    return _rng.choice(pool)


def substitute_synonyms(
    text: str,
    swap_rate: float = 0.35,
    rng: Optional[random.Random] = None,
) -> str:
    """Replace a fraction of adjectives and adverbs with WordNet synonyms.

    Only adjectives (JJ*) and adverbs (RB*) are eligible so that nouns,
    verbs, and named entities – which carry core logical meaning – are never
    modified.

    Args:
        text: Input sentence or paragraph.
        swap_rate: Fraction of eligible words to swap (0–1).
        rng: Optional seeded :class:`random.Random` for reproducible output.

    Returns:
        Text with some adjectives/adverbs substituted by synonyms.
        The original text is returned unchanged if NLTK data is unavailable.
    """
    if not _ensure_nltk_data():
        return text

    _rng = rng or random

    try:
        words = word_tokenize(text)
        tags = nltk.pos_tag(words)
    except Exception:
        return text

    result_words: List[str] = []
    for word, tag in tags:
        if tag in _SWAP_TAGS and word.isalpha() and len(word) > 0 and _rng.random() < swap_rate:
            synonym = _get_synonym(word, rng=_rng)
            if synonym:
                if word[0].isupper():
                    result_words.append(synonym.capitalize())
                else:
                    result_words.append(synonym)
                continue
        result_words.append(word)

    reconstructed = " ".join(result_words)
    # Fix spaces before punctuation introduced by word_tokenize
    reconstructed = re.sub(r"\s+([?.!,;:'\"])", r"\1", reconstructed)
    return reconstructed


# ---------------------------------------------------------------------------
# Capitalisation helper
# ---------------------------------------------------------------------------


def _capitalize_sentences(sentences: List[str]) -> List[str]:
    """Ensure every sentence starts with an uppercase letter."""
    out = []
    for s in sentences:
        s = s.strip()
        if s:
            out.append(s[0].upper() + s[1:] if len(s) > 1 else s[0].upper())
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def humanize(
    text: str,
    synonym_rate: float = 0.35,
    merge_rate: float = 0.25,
    seed: Optional[int] = None,
) -> HumanizeResult:
    """Transform AI-generated text to read as more human-written.

    The pipeline applies four transformations in sequence:

    1. **Marker stripping** – removes common AI transition phrases.
    2. **Sentence tokenisation** – splits the cleaned text into sentences.
    3. **Burstiness variation** – randomly merges adjacent sentences so that
       sentence lengths vary (a key human-writing trait).
    4. **Synonym substitution** – replaces a fraction of adjectives/adverbs
       with WordNet synonyms to raise lexical perplexity.

    Args:
        text: The input text to humanize (AI-generated or otherwise).
        synonym_rate: Fraction of eligible adjectives/adverbs to replace
            with synonyms (0–1). Default is 0.35.
        merge_rate: Probability of merging two consecutive sentences (0–1).
            Default is 0.25.
        seed: Optional integer seed for reproducible output across runs.

    Returns:
        A :class:`HumanizeResult` dataclass containing the humanized text
        and metadata (word counts, markers removed, sentences merged).
    """
    if not text or not text.strip():
        return HumanizeResult(
            original_text=text,
            humanized_text=text,
            original_word_count=0,
            humanized_word_count=0,
            markers_removed=0,
            sentences_merged=0,
        )

    rng = random.Random(seed)
    original_word_count = len(text.split())

    # Step 1 – strip AI markers
    cleaned, markers_removed = strip_ai_markers(text)

    if not cleaned.strip():
        # Edge case: entire text consisted of AI markers
        return HumanizeResult(
            original_text=text,
            humanized_text=text,
            original_word_count=original_word_count,
            humanized_word_count=original_word_count,
            markers_removed=markers_removed,
            sentences_merged=0,
        )

    # Step 2 – tokenise into sentences
    sentences = _tokenize_sentences(cleaned)

    if not sentences:
        return HumanizeResult(
            original_text=text,
            humanized_text=cleaned,
            original_word_count=original_word_count,
            humanized_word_count=len(cleaned.split()),
            markers_removed=markers_removed,
            sentences_merged=0,
        )

    # Step 3 – vary sentence lengths (burstiness)
    varied, sentences_merged = vary_sentence_lengths(sentences, merge_rate=merge_rate, rng=rng)

    # Step 4 – synonym substitution (perplexity)
    humanized_sentences = [
        substitute_synonyms(sent, swap_rate=synonym_rate, rng=rng) for sent in varied
    ]

    # Capitalise and join
    humanized_sentences = _capitalize_sentences(humanized_sentences)
    humanized_text = " ".join(humanized_sentences)
    humanized_word_count = len(humanized_text.split())

    return HumanizeResult(
        original_text=text,
        humanized_text=humanized_text,
        original_word_count=original_word_count,
        humanized_word_count=humanized_word_count,
        markers_removed=markers_removed,
        sentences_merged=sentences_merged,
    )
