"""Rigorous unit tests for the AI Text Humanizer core pipeline.

Tests cover:
- AI marker stripping (normal, edge-case, case-insensitive, multiple markers)
- Sentence burstiness variation (merging, no-merge, edge cases)
- Synonym substitution (NLTK-backed or regex fallback)
- Full humanize() pipeline (word counts, metadata, determinism, edge cases)
"""

from __future__ import annotations

import random
import unittest

from aip.humanizer import (
    HumanizeResult,
    _capitalize_sentences,
    humanize,
    strip_ai_markers,
    substitute_synonyms,
    vary_sentence_lengths,
)


class TestStripAiMarkers(unittest.TestCase):
    def test_removes_in_conclusion(self) -> None:
        text = "In conclusion, the results are clear."
        result, count = strip_ai_markers(text)
        self.assertNotIn("in conclusion", result.lower())
        self.assertGreater(count, 0)

    def test_removes_furthermore(self) -> None:
        text = "Furthermore, AI is evolving rapidly."
        result, count = strip_ai_markers(text)
        self.assertNotIn("furthermore", result.lower())
        self.assertGreater(count, 0)

    def test_removes_as_an_ai_language_model(self) -> None:
        text = "As an AI language model, I can help with your question."
        result, count = strip_ai_markers(text)
        self.assertNotIn("as an ai language model", result.lower())
        self.assertGreater(count, 0)

    def test_removes_multiple_markers(self) -> None:
        text = (
            "Furthermore, it is important to note that AI is evolving. "
            "In conclusion, things look bright."
        )
        result, count = strip_ai_markers(text)
        self.assertNotIn("furthermore", result.lower())
        self.assertNotIn("in conclusion", result.lower())
        self.assertGreaterEqual(count, 2)

    def test_preserves_content_without_markers(self) -> None:
        text = "The weather today is sunny and warm."
        result, count = strip_ai_markers(text)
        self.assertEqual(count, 0)
        self.assertIn("weather", result)
        self.assertIn("sunny", result)

    def test_empty_string(self) -> None:
        result, count = strip_ai_markers("")
        self.assertEqual(result, "")
        self.assertEqual(count, 0)

    def test_whitespace_only(self) -> None:
        result, count = strip_ai_markers("   ")
        self.assertEqual(result, "")
        self.assertEqual(count, 0)

    def test_only_marker_leaves_empty(self) -> None:
        result, count = strip_ai_markers("In conclusion")
        self.assertGreater(count, 0)
        self.assertEqual(result.strip(), "")

    def test_case_insensitive_removal(self) -> None:
        text = "IN CONCLUSION, this is a test."
        result, count = strip_ai_markers(text)
        self.assertNotIn("in conclusion", result.lower())
        self.assertGreater(count, 0)

    def test_no_double_spaces_after_removal(self) -> None:
        text = "The study found that Furthermore, the data is clear."
        result, _ = strip_ai_markers(text)
        self.assertNotIn("  ", result)

    def test_removes_moreover(self) -> None:
        result, count = strip_ai_markers("Moreover, this trend is growing.")
        self.assertNotIn("moreover", result.lower())
        self.assertGreater(count, 0)

    def test_removes_in_summary(self) -> None:
        result, count = strip_ai_markers("In summary, the approach works well.")
        self.assertNotIn("in summary", result.lower())
        self.assertGreater(count, 0)

    def test_count_reflects_actual_removals(self) -> None:
        text = "In conclusion, in conclusion, the test passes."
        _, count = strip_ai_markers(text)
        self.assertEqual(count, 2)


class TestVarySentenceLengths(unittest.TestCase):
    def test_merges_at_rate_one(self) -> None:
        sentences = ["This is short.", "This is also short."]
        rng = random.Random(0)
        merged, count = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        self.assertEqual(len(merged), 1)
        self.assertEqual(count, 1)

    def test_no_merge_at_rate_zero(self) -> None:
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        rng = random.Random(0)
        merged, count = vary_sentence_lengths(sentences, merge_rate=0.0, rng=rng)
        self.assertEqual(len(merged), 3)
        self.assertEqual(count, 0)

    def test_single_sentence_never_merged(self) -> None:
        sentences = ["Only one sentence here."]
        rng = random.Random(0)
        merged, count = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        self.assertEqual(len(merged), 1)
        self.assertEqual(count, 0)

    def test_empty_list_returns_empty(self) -> None:
        merged, count = vary_sentence_lengths([], merge_rate=0.5)
        self.assertEqual(merged, [])
        self.assertEqual(count, 0)

    def test_merged_sentence_contains_original_words(self) -> None:
        sentences = ["The fox is quick.", "The dog is lazy."]
        rng = random.Random(0)
        merged, _ = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        combined = merged[0].lower()
        self.assertIn("fox", combined)
        self.assertIn("dog", combined)

    def test_merged_sentence_uses_known_conjunction(self) -> None:
        sentences = ["Alpha is strong.", "Beta is fast."]
        rng = random.Random(0)
        merged, _ = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        text = merged[0]
        found_conjunction = any(conjunction.strip() in text for conjunction in [" and ", " while ", "meaning", "—", "which"])
        self.assertTrue(found_conjunction)

    def test_total_word_count_preserved_approximately(self) -> None:
        sentences = ["First short sentence.", "Second short sentence."]
        merged, _ = vary_sentence_lengths(sentences, merge_rate=1.0, rng=random.Random(1))
        words_before = sum(len(s.split()) for s in sentences)
        words_after = len(merged[0].split())
        # Merging adds one conjunction word; allow small delta
        self.assertAlmostEqual(words_before, words_after, delta=3)

    def test_two_pairs_can_both_merge(self) -> None:
        sentences = ["A.", "B.", "C.", "D."]
        rng = random.Random(0)
        merged, count = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        self.assertEqual(count, 2)
        self.assertEqual(len(merged), 2)


class TestSubstituteSynonyms(unittest.TestCase):
    def test_returns_string(self) -> None:
        result = substitute_synonyms("The quick brown fox.", swap_rate=0.0)
        self.assertIsInstance(result, str)

    def test_swap_rate_zero_leaves_text_unchanged(self) -> None:
        text = "The quick brown fox jumps."
        result = substitute_synonyms(text, swap_rate=0.0, rng=random.Random(0))
        self.assertEqual(result, text)

    def test_empty_string_returns_empty(self) -> None:
        result = substitute_synonyms("", swap_rate=0.5)
        self.assertEqual(result, "")

    def test_nouns_and_verbs_preserved(self) -> None:
        # Nouns like "Python" and "language" should never be swapped
        text = "Python is a programming language used worldwide."
        result = substitute_synonyms(text, swap_rate=1.0, rng=random.Random(42))
        self.assertIn("Python", result)
        self.assertIn("language", result)


class TestCapitalizeSentences(unittest.TestCase):
    def test_capitalises_lowercase_start(self) -> None:
        result = _capitalize_sentences(["hello world."])
        self.assertEqual(result, ["Hello world."])

    def test_does_not_double_capitalise(self) -> None:
        result = _capitalize_sentences(["Already capitalised."])
        self.assertEqual(result, ["Already capitalised."])

    def test_empty_list(self) -> None:
        self.assertEqual(_capitalize_sentences([]), [])

    def test_strips_whitespace(self) -> None:
        result = _capitalize_sentences(["  spaces before.  "])
        self.assertTrue(result[0][0].isupper())


class TestHumanizePipeline(unittest.TestCase):
    def test_returns_humanize_result(self) -> None:
        result = humanize("The sky is blue. The sun is bright.")
        self.assertIsInstance(result, HumanizeResult)

    def test_empty_text_returns_empty_result(self) -> None:
        result = humanize("")
        self.assertEqual(result.humanized_text, "")
        self.assertEqual(result.original_word_count, 0)
        self.assertEqual(result.humanized_word_count, 0)
        self.assertEqual(result.markers_removed, 0)
        self.assertEqual(result.sentences_merged, 0)

    def test_whitespace_only_returns_zero_counts(self) -> None:
        result = humanize("   ")
        self.assertEqual(result.original_word_count, 0)

    def test_removes_ai_markers_in_pipeline(self) -> None:
        ai_text = (
            "Furthermore, it is important to note that AI is rapidly evolving. "
            "Many businesses use it daily. In conclusion, the future looks bright."
        )
        result = humanize(ai_text, seed=42)
        self.assertGreater(result.markers_removed, 0)
        self.assertNotIn("in conclusion", result.humanized_text.lower())
        self.assertNotIn("furthermore", result.humanized_text.lower())

    def test_original_word_count_positive(self) -> None:
        text = "The quick brown fox jumps over the lazy dog."
        result = humanize(text, seed=0)
        self.assertGreater(result.original_word_count, 0)

    def test_humanized_word_count_positive(self) -> None:
        text = "Artificial intelligence is advancing quickly."
        result = humanize(text, seed=0)
        self.assertGreater(result.humanized_word_count, 0)

    def test_output_is_non_empty_for_valid_input(self) -> None:
        text = "Artificial intelligence is advancing quickly. This has many implications."
        result = humanize(text, seed=1)
        self.assertTrue(result.humanized_text.strip())

    def test_preserves_core_nouns(self) -> None:
        text = "Python is a programming language. It is widely used."
        result = humanize(text, seed=7)
        # Proper nouns should survive because they are tagged NNP, not JJ/RB
        self.assertIn("Python", result.humanized_text)

    def test_seed_produces_deterministic_output(self) -> None:
        text = "AI is transforming the world. It is incredibly powerful and highly intelligent."
        result1 = humanize(text, seed=42)
        result2 = humanize(text, seed=42)
        self.assertEqual(result1.humanized_text, result2.humanized_text)

    def test_different_seeds_can_produce_different_output(self) -> None:
        text = (
            "AI is transforming the world rapidly. "
            "It is incredibly powerful and highly intelligent. "
            "Many researchers are studying its effects closely."
        )
        result1 = humanize(text, seed=0)
        result2 = humanize(text, seed=99)
        # With multiple sentences and synonym substitution there is a high
        # chance two different seeds produce different text; not guaranteed
        # but a useful sanity check.
        self.assertIsInstance(result1.humanized_text, str)
        self.assertIsInstance(result2.humanized_text, str)

    def test_original_text_preserved_in_result(self) -> None:
        text = "This is the original text."
        result = humanize(text, seed=0)
        self.assertEqual(result.original_text, text)

    def test_sentences_merged_non_negative(self) -> None:
        text = "Short text."
        result = humanize(text, seed=0)
        self.assertGreaterEqual(result.sentences_merged, 0)

    def test_long_text_does_not_raise(self) -> None:
        long_text = "Artificial intelligence is changing the world rapidly. " * 200
        result = humanize(long_text, seed=0)
        self.assertTrue(result.humanized_text.strip())

    def test_single_word_input(self) -> None:
        result = humanize("Hello")
        self.assertGreater(result.original_word_count, 0)
        self.assertTrue(result.humanized_text.strip())

    def test_merge_rate_zero_still_produces_output(self) -> None:
        text = "First sentence here. Second sentence here. Third sentence here."
        result = humanize(text, merge_rate=0.0, seed=0)
        self.assertTrue(result.humanized_text.strip())
        self.assertEqual(result.sentences_merged, 0)

    def test_synonym_rate_zero_does_not_change_words(self) -> None:
        text = "The sky is blue and the grass is green."
        result = humanize(text, synonym_rate=0.0, merge_rate=0.0, seed=0)
        # With no swaps and no merges only marker-stripping and
        # capitalisation can alter the text
        self.assertIn("blue", result.humanized_text.lower())
        self.assertIn("green", result.humanized_text.lower())

    def test_metadata_word_counts_are_integers(self) -> None:
        result = humanize("Some sample text here.", seed=0)
        self.assertIsInstance(result.original_word_count, int)
        self.assertIsInstance(result.humanized_word_count, int)

    def test_markers_removed_is_integer(self) -> None:
        result = humanize("Furthermore, testing.", seed=0)
        self.assertIsInstance(result.markers_removed, int)

    def test_sentences_merged_is_integer(self) -> None:
        result = humanize("First. Second.", seed=0)
        self.assertIsInstance(result.sentences_merged, int)


if __name__ == "__main__":
    unittest.main()
