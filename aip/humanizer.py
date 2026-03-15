"""AI Text Humanizer – core NLP transformation pipeline.

The pipeline applies four transformations in order:

1. **Marker stripping** – removes common AI transition phrases
   (e.g. "In conclusion,", "Furthermore,", "As an AI language model").
2. **Sentence tokenisation** – splits the cleaned text into individual sentences.
3. **Burstiness variation** – randomly merges adjacent sentences using em-dashes
   (—), semicolons (;), and conjunctions so that sentence lengths vary, a key
   trait of human writing.
4. **Contraction manipulation** – randomly expands or contracts common English
   phrases (e.g. "do not" → "don't") to alter token count and perplexity
   without changing semantic meaning.

When ``adversarial_mode=True`` is passed to :func:`humanize`, two additional
evasion techniques are applied:

5. **Zero-width space injection** – inserts invisible U+200B characters inside
   long words to break AI-detector tokenisers (GPTZero, Turnitin, etc.).
6. **Homoglyph swapping** – replaces a small fraction of Latin characters with
   visually identical Cyrillic lookalikes, making tokens unrecognisable to
   detector models while appearing completely normal to human readers.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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

# Conjunctions used when merging two adjacent sentences.
# Em-dashes (—) and semicolons (;) increase burstiness variety.
_MERGE_CONJUNCTIONS: List[str] = [
    " and ",
    " while ",
    ", meaning ",
    " — ",
    ", which ",
    "; ",
]

# ---------------------------------------------------------------------------
# Contraction manipulation tables
# ---------------------------------------------------------------------------

# Expand: contraction → full form  (e.g. "it's" → "it is")
_EXPAND_CONTRACTIONS: List[Tuple[str, str]] = [
    (r"\bit's\b", "it is"),
    (r"\bI'm\b", "I am"),
    (r"\bdon't\b", "do not"),
    (r"\bdoesn't\b", "does not"),
    (r"\bcan't\b", "cannot"),
    (r"\bwon't\b", "will not"),
    (r"\bwe're\b", "we are"),
    (r"\bthey're\b", "they are"),
    (r"\bI've\b", "I have"),
    (r"\bI'll\b", "I will"),
    (r"\bwouldn't\b", "would not"),
    (r"\bcouldn't\b", "could not"),
    (r"\bshouldn't\b", "should not"),
    (r"\bisn't\b", "is not"),
    (r"\baren't\b", "are not"),
    (r"\bwasn't\b", "was not"),
    (r"\bweren't\b", "were not"),
    (r"\bhadn't\b", "had not"),
    (r"\bhasn't\b", "has not"),
    (r"\bhaven't\b", "have not"),
    (r"\bdidn't\b", "did not"),
]

# Contract: full form → contraction  (e.g. "do not" → "don't")
_CONTRACT_PHRASES: List[Tuple[str, str]] = [
    (r"\bit is\b", "it's"),
    (r"\bI am\b", "I'm"),
    (r"\bdo not\b", "don't"),
    (r"\bdoes not\b", "doesn't"),
    (r"\bcannot\b", "can't"),
    (r"\bwill not\b", "won't"),
    (r"\bwe are\b", "we're"),
    (r"\bthey are\b", "they're"),
    (r"\bI have\b", "I've"),
    (r"\bI will\b", "I'll"),
    (r"\bwould not\b", "wouldn't"),
    (r"\bcould not\b", "couldn't"),
    (r"\bshould not\b", "shouldn't"),
    (r"\bis not\b", "isn't"),
    (r"\bare not\b", "aren't"),
    (r"\bwas not\b", "wasn't"),
    (r"\bwere not\b", "weren't"),
    (r"\bhad not\b", "hadn't"),
    (r"\bhas not\b", "hasn't"),
    (r"\bhave not\b", "haven't"),
    (r"\bdid not\b", "didn't"),
]

# ---------------------------------------------------------------------------
# Homoglyph map: Latin character → visually identical Cyrillic character
# Only pairs confirmed to be visually indistinguishable in common fonts.
# ---------------------------------------------------------------------------

_HOMOGLYPH_MAP: Dict[str, str] = {
    "a": "\u0430",  # Latin 'a' (U+0061) → Cyrillic 'а' (U+0430)
    "e": "\u0435",  # Latin 'e' (U+0065) → Cyrillic 'е' (U+0435)
    "o": "\u043E",  # Latin 'o' (U+006F) → Cyrillic 'о' (U+043E)
    "p": "\u0440",  # Latin 'p' (U+0070) → Cyrillic 'р' (U+0440)
    "c": "\u0441",  # Latin 'c' (U+0063) → Cyrillic 'с' (U+0441)
    "x": "\u0445",  # Latin 'x' (U+0078) → Cyrillic 'х' (U+0445)
    "A": "\u0410",  # Latin 'A' (U+0041) → Cyrillic 'А' (U+0410)
    "B": "\u0412",  # Latin 'B' (U+0042) → Cyrillic 'В' (U+0412)
    "C": "\u0421",  # Latin 'C' (U+0043) → Cyrillic 'С' (U+0421)
    "E": "\u0415",  # Latin 'E' (U+0045) → Cyrillic 'Е' (U+0415)
    "H": "\u041D",  # Latin 'H' (U+0048) → Cyrillic 'Н' (U+041D)
    "K": "\u041A",  # Latin 'K' (U+004B) → Cyrillic 'К' (U+041A)
    "M": "\u041C",  # Latin 'M' (U+004D) → Cyrillic 'М' (U+041C)
    "O": "\u041E",  # Latin 'O' (U+004F) → Cyrillic 'О' (U+041E)
    "P": "\u0420",  # Latin 'P' (U+0050) → Cyrillic 'Р' (U+0420)
    "T": "\u0422",  # Latin 'T' (U+0054) → Cyrillic 'Т' (U+0422)
    "X": "\u0425",  # Latin 'X' (U+0058) → Cyrillic 'Х' (U+0425)
}

# Minimum word length eligible for zero-width space injection
_ZWS_MIN_WORD_LEN = 6

# Invisible zero-width space character (U+200B)
_ZWSP = "\u200B"

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
    """Split *text* into sentences using a regex-based approach.

    Splits on ``.``, ``!``, and ``?`` followed by whitespace, which handles
    the vast majority of prose without requiring any external NLP library.
    """
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
    conjunction, em-dash (—), or semicolon (;) makes the text read more
    naturally and raises the burstiness score used by AI detectors.

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
            tail = next_sent.lstrip()
            # Semicolons keep the following clause capitalised; other
            # conjunctions lower-case the first letter for natural flow.
            if conjunction != "; " and len(tail) > 1:
                tail = tail[0].lower() + tail[1:]
            elif conjunction != "; " and tail:
                tail = tail[0].lower()
            result.append(base + conjunction + tail)
            merges += 1
            i += 2
        else:
            result.append(sentences[i])
            i += 1

    return result, merges


# ---------------------------------------------------------------------------
# Step 4 – Contraction manipulation (token-count / perplexity alteration)
# ---------------------------------------------------------------------------


def apply_contractions(
    text: str,
    rng: Optional[random.Random] = None,
) -> str:
    """Randomly expand or contract common English phrases.

    On each call the function randomly decides to either expand contractions
    (e.g. "it's" → "it is") or contract expanded forms (e.g. "do not" →
    "don't").  Each individual substitution is applied with ~50 % probability
    so that only a subset of matches is changed, producing natural variation.

    Altering the token count and vocabulary patterns this way changes the
    mathematical perplexity profile of the text without modifying its meaning.

    Args:
        text: Input sentence or paragraph.
        rng: Optional seeded :class:`random.Random` for reproducible output.

    Returns:
        Text with some contractions expanded or contracted.
    """
    _rng = rng or random
    table = _EXPAND_CONTRACTIONS if _rng.random() < 0.5 else _CONTRACT_PHRASES
    result = text
    for pattern, replacement in table:
        if _rng.random() < 0.5:
            result = re.sub(pattern, replacement, result)
    return result


# ---------------------------------------------------------------------------
# Adversarial evasion helpers (Steps 5 & 6)
# ---------------------------------------------------------------------------


def inject_zero_width_spaces(
    text: str,
    rate: float = 0.15,
    rng: Optional[random.Random] = None,
) -> str:
    """Inject invisible zero-width spaces (U+200B) inside long words.

    Words shorter than ``_ZWS_MIN_WORD_LEN`` characters are skipped.  Each
    eligible word has a ZWSP inserted at a random interior position with
    probability ``rate``.

    The resulting text is visually identical to the original for human readers
    but breaks the byte-pair encoding (BPE) tokenisers used by AI detectors
    such as GPTZero and Turnitin, causing them to misidentify word tokens and
    dramatically lower their AI-detection confidence score.

    Args:
        text: Input text.
        rate: Fraction of eligible long words to receive a ZWSP (0–1).
        rng: Optional seeded :class:`random.Random` for reproducible output.

    Returns:
        Text with invisible zero-width spaces injected into some long words.
    """
    _rng = rng or random

    def _inject_word(match: re.Match) -> str:  # type: ignore[type-arg]
        word = match.group(0)
        if len(word) >= _ZWS_MIN_WORD_LEN and _rng.random() < rate:
            pos = _rng.randint(1, len(word) - 1)
            return word[:pos] + _ZWSP + word[pos:]
        return word

    return re.sub(r"[A-Za-z]+", _inject_word, text)


def apply_homoglyph_swaps(
    text: str,
    rate: float = 0.03,
    rng: Optional[random.Random] = None,
) -> str:
    """Replace a small fraction of Latin characters with Cyrillic homoglyphs.

    Each character present in ``_HOMOGLYPH_MAP`` is replaced with its
    visually identical Cyrillic counterpart with probability ``rate``
    (default 3 %).  The substituted characters look identical to the originals
    in all common fonts, so the text appears perfectly normal to human readers
    while the raw bytes confuse AI-detector tokenisers.

    Args:
        text: Input text.
        rate: Per-character replacement probability (0–1).  Keep this low
            (2–5 %) to preserve search-ability and copy-paste behaviour.
        rng: Optional seeded :class:`random.Random` for reproducible output.

    Returns:
        Text with a small fraction of Latin letters replaced by Cyrillic
        lookalikes.
    """
    _rng = rng or random
    result: List[str] = []
    for ch in text:
        cyrillic = _HOMOGLYPH_MAP.get(ch)
        if cyrillic is not None and _rng.random() < rate:
            result.append(cyrillic)
        else:
            result.append(ch)
    return "".join(result)


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
    merge_rate: float = 0.25,
    seed: Optional[int] = None,
    adversarial_mode: bool = False,
    adversarial_zws_rate: float = 0.15,
    adversarial_homoglyph_rate: float = 0.03,
) -> HumanizeResult:
    """Transform AI-generated text to read as more human-written.

    The pipeline applies four transformations in sequence:

    1. **Marker stripping** – removes common AI transition phrases.
    2. **Sentence tokenisation** – splits the cleaned text into sentences.
    3. **Burstiness variation** – randomly merges adjacent sentences using
       em-dashes (—), semicolons (;), and conjunctions so that sentence
       lengths vary (a key human-writing trait).
    4. **Contraction manipulation** – randomly expands or contracts common
       phrases (e.g. "it is" ↔ "it's") to alter token count and perplexity
       without changing semantic meaning.

    When ``adversarial_mode`` is ``True``, two additional evasion techniques
    are applied after the standard pipeline:

    5. **Zero-width space injection** – inserts invisible U+200B characters
       inside long words to break AI-detector tokenisers.
    6. **Homoglyph swapping** – replaces a small fraction of Latin characters
       with visually identical Cyrillic lookalikes to further confuse
       tokenisers without affecting human readability.

    Args:
        text: The input text to humanize (AI-generated or otherwise).
        merge_rate: Probability of merging two consecutive sentences (0–1).
            Default is 0.25.
        seed: Optional integer seed for reproducible output across runs.
        adversarial_mode: When ``True``, apply zero-width space injection and
            homoglyph swapping for maximum AI-detection evasion.
        adversarial_zws_rate: Fraction of eligible long words to receive a
            zero-width space injection (0–1). Only used when
            ``adversarial_mode=True``. Default is 0.15.
        adversarial_homoglyph_rate: Per-character homoglyph replacement
            probability (0–1). Only used when ``adversarial_mode=True``.
            Default is 0.03 (3 %).

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

    # Step 4 – contraction manipulation (token / perplexity variation)
    manipulated = [apply_contractions(sent, rng=rng) for sent in varied]

    # Capitalise and join
    manipulated = _capitalize_sentences(manipulated)
    humanized_text = " ".join(manipulated)

    # Steps 5 & 6 – adversarial evasion (optional)
    if adversarial_mode:
        humanized_text = inject_zero_width_spaces(
            humanized_text, rate=adversarial_zws_rate, rng=rng
        )
        humanized_text = apply_homoglyph_swaps(
            humanized_text, rate=adversarial_homoglyph_rate, rng=rng
        )

    humanized_word_count = len(humanized_text.split())

    return HumanizeResult(
        original_text=text,
        humanized_text=humanized_text,
        original_word_count=original_word_count,
        humanized_word_count=humanized_word_count,
        markers_removed=markers_removed,
        sentences_merged=sentences_merged,
    )
