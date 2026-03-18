"""AI Text Humanizer – core NLP transformation pipeline.

The pipeline applies eight transformations in order:

1. **Marker stripping** – removes common AI transition phrases
   (e.g. "In conclusion,", "Furthermore,", "As an AI language model").
2. **Sentence tokenisation** – splits the cleaned text into individual sentences.
3. **Contraction insertion** – converts formal expansions ("do not" → "don't",
   "it is" → "it's") to match natural human writing style.
4. **Clause reordering** – moves prepositional/adverbial phrases to vary
   subject-verb-object monotony.
5. **Sentence splitting** – breaks long compound sentences at coordinating
   conjunctions to create short punchy sentences.
6. **Burstiness variation** – randomly merges adjacent sentences so that
   sentence lengths vary, a key trait of human writing.
7. **Discourse filler insertion** – prepends natural human phrases like
   "Honestly,", "Look,", "The thing is," to some sentences.
8. **Synonym substitution** – replaces a fraction of adjectives and adverbs
   with WordNet synonyms to raise lexical perplexity without changing meaning.
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
    # Classic AI conclusions / transitions
    "in conclusion,",
    "in conclusion",
    "it is important to note that",
    "it is worth noting that",
    "it is crucial to understand that",
    "it should be noted that",
    "it is essential to note that",
    "it is worth mentioning that",
    "it is important to understand that",
    "it is imperative to note that",
    "as an ai language model,",
    "as an ai language model",
    "as an ai,",
    "as an ai",
    "i cannot browse",
    "i do not have real-time",
    # Transition / connector phrases
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
    # Modern GPT-era markers
    "it's worth noting that",
    "it's important to note that",
    "it's crucial to note that",
    "it is also worth noting that",
    "it is also important to note that",
    "in today's rapidly evolving",
    "in today's fast-paced",
    "in today's digital age,",
    "in today's digital age",
    "in the ever-evolving landscape of",
    "in this day and age,",
    "in this day and age",
    "the importance of this cannot be overstated",
    "this cannot be overstated",
    "this underscores the importance of",
    "plays a crucial role in",
    "plays a pivotal role in",
    "plays a vital role in",
    "serves as a testament to",
    "it's essential to recognize that",
    "delves into",
    "delve into",
    "it's also worth mentioning that",
    "on the other hand,",
    "having said that,",
    "that being said,",
    "with that being said,",
    "notwithstanding,",
    "henceforth,",
    "consequently,",
    "in light of this,",
    "in light of the above,",
    "as a result of this,",
    "it can be concluded that",
    "to conclude,",
    "to sum up,",
    "all things considered,",
    "taking everything into account,",
    "in essence,",
    "by and large,",
    "for the most part,",
    "on a broader note,",
    "from a broader perspective,",
    "in a nutshell,",
    "overall,",
]

# ---------------------------------------------------------------------------
# Contraction maps (formal → contracted)
# ---------------------------------------------------------------------------

_CONTRACTIONS: List[Tuple[str, str]] = [
    ("do not", "don't"),
    ("does not", "doesn't"),
    ("did not", "didn't"),
    ("is not", "isn't"),
    ("are not", "aren't"),
    ("was not", "wasn't"),
    ("were not", "weren't"),
    ("will not", "won't"),
    ("would not", "wouldn't"),
    ("could not", "couldn't"),
    ("should not", "shouldn't"),
    ("has not", "hasn't"),
    ("have not", "haven't"),
    ("had not", "hadn't"),
    ("can not", "can't"),
    ("cannot", "can't"),
    ("it is", "it's"),
    ("it has", "it's"),
    ("that is", "that's"),
    ("there is", "there's"),
    ("there are", "there're"),
    ("they are", "they're"),
    ("they have", "they've"),
    ("they will", "they'll"),
    ("they would", "they'd"),
    ("we are", "we're"),
    ("we have", "we've"),
    ("we will", "we'll"),
    ("we would", "we'd"),
    ("you are", "you're"),
    ("you have", "you've"),
    ("you will", "you'll"),
    ("you would", "you'd"),
    ("I am", "I'm"),
    ("I have", "I've"),
    ("I will", "I'll"),
    ("I would", "I'd"),
    ("he is", "he's"),
    ("he has", "he's"),
    ("he will", "he'll"),
    ("he would", "he'd"),
    ("she is", "she's"),
    ("she has", "she's"),
    ("she will", "she'll"),
    ("she would", "she'd"),
    ("who is", "who's"),
    ("who has", "who's"),
    ("what is", "what's"),
    ("what has", "what's"),
    ("let us", "let's"),
]

# ---------------------------------------------------------------------------
# Discourse fillers — natural human interjections
# ---------------------------------------------------------------------------

_DISCOURSE_FILLERS: List[str] = [
    "Honestly,",
    "Look,",
    "The thing is,",
    "Here's the deal —",
    "Basically,",
    "Truth is,",
    "I mean,",
    "Sure,",
    "Right,",
    "So yeah,",
    "In practice,",
    "Realistically,",
    "At the end of the day,",
    "For what it's worth,",
    "Point being,",
    "To be fair,",
    "If you think about it,",
    "No doubt,",
    "Funny enough,",
    "Interestingly,",
    "Naturally,",
    "And honestly,",
    "As it turns out,",
    "The reality is,",
    "Plain and simple,",
]

# POS tags eligible for synonym substitution (adjectives and adverbs only)
_SWAP_TAGS = {"JJ", "JJR", "JJS", "RB", "RBR", "RBS"}

# Words that must never be swapped – they carry precise contextual meaning that
# WordNet synonyms routinely mishandle.
_SYNONYM_BLACKLIST: frozenset = frozenset(
    {
        # Function words / adverbs that must stay exact
        "not", "very", "also", "just", "only", "even", "still", "never",
        "always", "often", "much", "more", "most", "less", "least",
        "well", "quite", "rather", "really", "too", "so", "now", "then",
        "here", "there", "where", "when", "how", "why", "yet", "ago",
        # Adjectives that WordNet maps to absurd synonyms
        "same", "social", "cloud", "public", "human", "local", "global",
        "digital", "data", "privacy", "ethical", "real", "true", "false",
        "artificial", "recent", "significant", "essential", "substantial",
        "particular", "continued", "important", "potential", "rapid",
        "various", "certain", "entire", "overall", "likely", "unlikely",
        "available", "possible", "impossible", "necessary", "relevant",
        "effective", "efficient", "successful", "traditional", "fundamental",
        "critical", "crucial", "vital", "key", "major", "minor",
        "daily", "annual", "initial", "final", "total", "original",
        "previous", "current", "future", "present", "past", "next",
        "many", "several", "numerous", "few", "other", "such",
        "able", "unable", "likely", "unlikely", "further", "additional",
    }
)

# Conjunctions used when merging two adjacent short sentences (expanded)
_MERGE_CONJUNCTIONS = [
    " and ",
    " while ",
    ", meaning ",
    " — ",
    ", which ",
    ", and ",
    "; ",
    " but ",
    " yet ",
    ", so ",
    " — and ",
    ", plus ",
    " (and ",
    ", essentially ",
    " — basically ",
    ", right? And ",
]

# Coordinating conjunctions at which long sentences can be split
_SPLIT_CONJUNCTIONS = [
    ", and ",
    ", but ",
    ", or ",
    ", yet ",
    ", so ",
    "; ",
    " because ",
    " although ",
    " however ",
    ", however,",
    " whereas ",
    " while ",
]

# Prepositional / adverbial openers that can be moved around
_MOVABLE_CLAUSE_PATTERNS = [
    # "In [year/time], ..." → move to end
    re.compile(r"^(In \d{4},?\s+)(.+)$", re.IGNORECASE),
    # "During the [noun], ..." → move to end
    re.compile(r"^(During the \w[\w\s]{0,30},?\s+)(.+)$", re.IGNORECASE),
    # "Over the past [X], ..." → move to end
    re.compile(r"^(Over the (?:past|last|next) [\w\s]{1,20},?\s+)(.+)$", re.IGNORECASE),
    # "According to [X], ..." → move to end
    re.compile(r"^(According to [\w\s]{1,40},?\s+)(.+)$", re.IGNORECASE),
    # "As a result, ..."
    re.compile(r"^(As a result,?\s+)(.+)$", re.IGNORECASE),
    # "For example, ..."
    re.compile(r"^(For (?:example|instance),?\s+)(.+)$", re.IGNORECASE),
    # "In particular, ..."
    re.compile(r"^(In particular,?\s+)(.+)$", re.IGNORECASE),
    # "On the other hand, ..."
    re.compile(r"^(On the other hand,?\s+)(.+)$", re.IGNORECASE),
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
# Step 3 – Contraction insertion
# ---------------------------------------------------------------------------


def insert_contractions(
    text: str,
    rate: float = 0.65,
    rng: Optional[random.Random] = None,
) -> str:
    """Convert formal expansions to contractions probabilistically.

    E.g. ``"do not"`` → ``"don't"``, ``"it is"`` → ``"it's"``.
    Contractions make text sound significantly more human.

    Args:
        text: Input text.
        rate: Probability (0–1) of contracting each match. Default 0.65.
        rng: Optional seeded RNG for reproducibility.

    Returns:
        Text with formal phrases probabilistically contracted.
    """
    if rate <= 0:
        return text

    _rng = rng or random

    result = text
    # Process longer patterns first to avoid partial match issues
    for formal, contracted in sorted(_CONTRACTIONS, key=lambda x: len(x[0]), reverse=True):
        # Build a case-insensitive pattern with word boundaries
        pattern = re.compile(r"\b" + re.escape(formal) + r"\b", re.IGNORECASE)

        def _replacer(match: re.Match) -> str:
            if _rng.random() < rate:
                original = match.group(0)
                # Preserve capitalisation of the first letter
                if original[0].isupper():
                    return contracted[0].upper() + contracted[1:]
                return contracted
            return match.group(0)

        result = pattern.sub(_replacer, result)

    return result


# ---------------------------------------------------------------------------
# Step 4 – Clause reordering
# ---------------------------------------------------------------------------


def reorder_clauses(
    sentences: List[str],
    rate: float = 0.20,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Move leading prepositional/adverbial phrases to end of sentence.

    AI text almost always starts with the main clause. Humans frequently
    put qualifiers at the end. This transform moves opening clauses
    (e.g. "In 2024, X happened" → "X happened in 2024") to break the
    monotonous AI structure.

    Args:
        sentences: List of individual sentences.
        rate: Probability of reordering each eligible sentence.
        rng: Optional seeded RNG.

    Returns:
        List of sentences with some clauses reordered.
    """
    if rate <= 0:
        return list(sentences)

    _rng = rng or random
    result = []

    for sent in sentences:
        if _rng.random() >= rate:
            result.append(sent)
            continue

        reordered = False
        for pattern in _MOVABLE_CLAUSE_PATTERNS:
            m = pattern.match(sent.strip())
            if m:
                clause = m.group(1).strip().rstrip(",")
                remainder = m.group(2).strip()
                # Capitalise remainder, lowercase the moved clause
                if remainder:
                    new_sent = remainder.rstrip(".!?") + ", " + clause.lower() + "."
                    # Ensure first letter is capitalised
                    new_sent = new_sent[0].upper() + new_sent[1:]
                    result.append(new_sent)
                    reordered = True
                    break

        if not reordered:
            result.append(sent)

    return result


# ---------------------------------------------------------------------------
# Step 5 – Sentence splitting
# ---------------------------------------------------------------------------


def split_long_sentences(
    sentences: List[str],
    rate: float = 0.30,
    min_words: int = 18,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Break long compound sentences into shorter ones.

    AI detectors look for uniformly medium-length sentences. Splitting some
    long sentences creates the short, punchy fragments that characterise
    human writing.

    Args:
        sentences: List of sentences to process.
        rate: Probability of splitting each eligible sentence.
        min_words: Minimum word count for a sentence to be eligible.
        rng: Optional seeded RNG.

    Returns:
        List of sentences, potentially longer than input due to splits.
    """
    if rate <= 0:
        return list(sentences)

    _rng = rng or random
    result: List[str] = []

    for sent in sentences:
        word_count = len(sent.split())
        if word_count < min_words or _rng.random() >= rate:
            result.append(sent)
            continue

        # Try to split at a conjunction
        split_done = False
        # Shuffle conjunction order for variety
        conjs = list(_SPLIT_CONJUNCTIONS)
        _rng.shuffle(conjs)
        for conj in conjs:
            idx = sent.lower().find(conj.lower())
            if idx > 0:
                left = sent[:idx].strip()
                right_start = idx + len(conj)
                right = sent[right_start:].strip()

                # Only split if both halves are substantial
                if len(left.split()) >= 5 and len(right.split()) >= 4:
                    # Ensure left ends with punctuation
                    if not left[-1] in ".!?":
                        left = left.rstrip(",;") + "."
                    # Capitalise right side
                    if right:
                        right = right[0].upper() + right[1:]
                        if not right[-1] in ".!?":
                            right = right.rstrip(",;") + "."
                    result.append(left)
                    result.append(right)
                    split_done = True
                    break

        if not split_done:
            result.append(sent)

    return result


# ---------------------------------------------------------------------------
# Step 6 – Burstiness variation (sentence merging) – enhanced
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
# Step 7 – Discourse filler insertion
# ---------------------------------------------------------------------------


def insert_discourse_fillers(
    sentences: List[str],
    rate: float = 0.08,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """Prepend natural human discourse fillers to some sentences.

    Humans frequently start sentences with "Honestly,", "Look,", "I mean,"
    etc. AI text almost never does. Adding these sparingly makes text
    sound significantly more natural.

    Args:
        sentences: List of sentences.
        rate: Probability of adding a filler to each sentence. Keep low
            (0.05–0.15) for realistic results.
        rng: Optional seeded RNG.

    Returns:
        List of sentences with some fillers prepended.
    """
    if rate <= 0 or not sentences:
        return list(sentences)

    _rng = rng or random
    result: List[str] = []
    used_fillers: set = set()

    for i, sent in enumerate(sentences):
        # Never add filler to the very first sentence or very short sentences
        if i == 0 or len(sent.split()) < 4 or _rng.random() >= rate:
            result.append(sent)
            continue

        # Pick a filler we haven't used yet (avoid repetition)
        available = [f for f in _DISCOURSE_FILLERS if f not in used_fillers]
        if not available:
            available = list(_DISCOURSE_FILLERS)
            used_fillers.clear()

        filler = _rng.choice(available)
        used_fillers.add(filler)

        # Lowercase the original sentence start and prepend filler
        body = sent.lstrip()
        if body and body[0].isupper():
            body = body[0].lower() + body[1:]

        result.append(f"{filler} {body}")

    return result


# ---------------------------------------------------------------------------
# Step 8 – Synonym substitution (perplexity raising) – improved
# ---------------------------------------------------------------------------


def _get_synonym(word: str, rng: Optional[random.Random] = None) -> Optional[str]:
    """Return a single WordNet synonym for *word*, or ``None`` if unavailable.

    Synonyms are drawn from the **first two** synsets (most common senses)
    to increase vocabulary diversity while staying safe. Words in
    ``_SYNONYM_BLACKLIST`` are never swapped. Single-character words and
    very short words (< 3 chars) are also skipped.
    """
    if not _NLTK_AVAILABLE:
        return None

    lower = word.lower()
    if lower in _SYNONYM_BLACKLIST:
        return None

    if len(word) < 3:
        return None

    synsets = wordnet.synsets(word)
    if not synsets:
        return None

    # Use up to the first 2 synsets (most frequent senses) to widen the pool
    # while avoiding truly obscure meanings
    synonyms: set = set()
    for synset in synsets[:2]:
        for lemma in synset.lemmas():
            name = lemma.name()
            if (
                name.lower() != lower
                and "_" not in name
                and name.isalpha()
                and len(name) >= 3
            ):
                synonyms.add(name)

    if not synonyms:
        return None

    # Prefer synonyms that don't share the same 3-character stem as the
    # original word, to maximise vocabulary diversity.
    stem_prefix = lower[:min(len(word), 3)]
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
        swap_rate: Fraction of eligible words to swap (0–1). No hard cap —
            quality is controlled per-word by the blacklist and synset
            filtering.
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
        if tag in _SWAP_TAGS and word.isalpha() and len(word) >= 3 and _rng.random() < swap_rate:
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
    # Fix NLTK tokenizer splitting contractions (e.g. "ca n't" → "can't")
    reconstructed = re.sub(r"\b(ca)\s+(n't)", r"\1\2", reconstructed)
    reconstructed = re.sub(r"\b(wo)\s+(n't)", r"\1\2", reconstructed)
    reconstructed = re.sub(r"\b(sha)\s+(n't)", r"\1\2", reconstructed)
    reconstructed = re.sub(r"(\w)\s+('(?:s|re|ve|ll|d|t|m))\b", r"\1\2", reconstructed)
    reconstructed = re.sub(r"\b(n)\s+('t)\b", r"\1\2", reconstructed)
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
    contraction_rate: float = 0.65,
    clause_reorder_rate: float = 0.20,
    split_rate: float = 0.30,
    filler_rate: float = 0.08,
    seed: Optional[int] = None,
) -> HumanizeResult:
    """Transform AI-generated text to read as more human-written.

    The pipeline applies eight transformations in sequence:

    1. **Marker stripping** – removes common AI transition phrases.
    2. **Sentence tokenisation** – splits the cleaned text into sentences.
    3. **Contraction insertion** – converts formal expansions to contractions.
    4. **Clause reordering** – varies sentence structure by moving clauses.
    5. **Sentence splitting** – breaks long compound sentences.
    6. **Burstiness variation** – randomly merges adjacent sentences.
    7. **Discourse fillers** – inserts natural human interjections.
    8. **Synonym substitution** – replaces adjectives/adverbs with synonyms.

    Args:
        text: The input text to humanize (AI-generated or otherwise).
        synonym_rate: Fraction of eligible adjectives/adverbs to replace
            with synonyms (0–1). Default is 0.35.
        merge_rate: Probability of merging two consecutive sentences (0–1).
            Default is 0.25.
        contraction_rate: Probability of contracting formal phrases (0–1).
            Default is 0.65.
        clause_reorder_rate: Probability of reordering clauses in eligible
            sentences (0–1). Default is 0.20.
        split_rate: Probability of splitting long compound sentences (0–1).
            Default is 0.30.
        filler_rate: Probability of inserting discourse fillers (0–1).
            Default is 0.08.
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

    # Step 3 – insert contractions (applied to full text, then re-tokenise)
    contracted_text = insert_contractions(
        " ".join(sentences), rate=contraction_rate, rng=rng
    )
    sentences = _tokenize_sentences(contracted_text)
    if not sentences:
        sentences = [contracted_text]

    # Step 4 – clause reordering
    sentences = reorder_clauses(sentences, rate=clause_reorder_rate, rng=rng)

    # Step 5 – sentence splitting
    sentences = split_long_sentences(sentences, rate=split_rate, rng=rng)

    # Step 6 – vary sentence lengths (burstiness)
    varied, sentences_merged = vary_sentence_lengths(
        sentences, merge_rate=merge_rate, rng=rng
    )

    # Step 7 – discourse filler insertion
    varied = insert_discourse_fillers(varied, rate=filler_rate, rng=rng)

    # Step 8 – synonym substitution (perplexity)
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
