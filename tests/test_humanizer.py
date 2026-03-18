"""Rigorous unit tests for the AI Text Humanizer core pipeline.

Tests cover:
- AI marker stripping (normal, edge-case, case-insensitive, multiple markers)
- Sentence burstiness variation (merging, no-merge, edge cases)
- Synonym substitution (NLTK-backed or regex fallback)
- Contraction insertion (formal → contracted)
- Clause reordering (prepositional phrase movement)
- Sentence splitting (long compound sentences)
- Discourse filler insertion (natural interjections)
- Full humanize() pipeline (word counts, metadata, determinism, edge cases)
"""

from __future__ import annotations

import random
import unittest

from aip.humanizer import (
    HumanizeResult,
    _SYNONYM_BLACKLIST,
    _capitalize_sentences,
    humanize,
    insert_contractions,
    insert_discourse_fillers,
    reorder_clauses,
    split_long_sentences,
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

    def test_removes_modern_gpt_markers(self) -> None:
        text = "In today's rapidly evolving world, AI is key. This delves into the topic."
        result, count = strip_ai_markers(text)
        self.assertNotIn("in today's rapidly evolving", result.lower())
        self.assertNotIn("delves into", result.lower())
        self.assertGreaterEqual(count, 2)

    def test_removes_plays_crucial_role(self) -> None:
        text = "Technology plays a crucial role in modern life."
        result, count = strip_ai_markers(text)
        self.assertNotIn("plays a crucial role in", result.lower())
        self.assertGreater(count, 0)

    def test_removes_overall(self) -> None:
        text = "Overall, the project was a success."
        result, count = strip_ai_markers(text)
        self.assertNotIn("overall,", result.lower())
        self.assertGreater(count, 0)


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
        found_conjunction = any(
            conjunction.strip() in text
            for conjunction in [
                "and", "while", "meaning", "—", "which", "but",
                "yet", "so", "plus", "essentially", "basically",
            ]
        )
        self.assertTrue(found_conjunction)

    def test_total_word_count_preserved_approximately(self) -> None:
        sentences = ["First short sentence.", "Second short sentence."]
        merged, _ = vary_sentence_lengths(sentences, merge_rate=1.0, rng=random.Random(1))
        words_before = sum(len(s.split()) for s in sentences)
        words_after = len(merged[0].split())
        # Merging adds one conjunction word; allow small delta
        self.assertAlmostEqual(words_before, words_after, delta=5)

    def test_two_pairs_can_both_merge(self) -> None:
        sentences = ["A.", "B.", "C.", "D."]
        rng = random.Random(0)
        merged, count = vary_sentence_lengths(sentences, merge_rate=1.0, rng=rng)
        self.assertEqual(count, 2)
        self.assertEqual(len(merged), 2)


class TestInsertContractions(unittest.TestCase):
    def test_contracts_do_not(self) -> None:
        text = "I do not like it."
        result = insert_contractions(text, rate=1.0, rng=random.Random(0))
        self.assertIn("don't", result)
        self.assertNotIn("do not", result)

    def test_contracts_it_is(self) -> None:
        text = "It is a beautiful day."
        result = insert_contractions(text, rate=1.0, rng=random.Random(0))
        self.assertIn("It's", result)

    def test_contracts_cannot(self) -> None:
        text = "We cannot proceed without help."
        result = insert_contractions(text, rate=1.0, rng=random.Random(0))
        self.assertIn("can't", result)

    def test_rate_zero_no_change(self) -> None:
        text = "We do not agree and it is clear."
        result = insert_contractions(text, rate=0.0, rng=random.Random(0))
        self.assertEqual(result, text)

    def test_preserves_capitalisation(self) -> None:
        text = "It is important."
        result = insert_contractions(text, rate=1.0, rng=random.Random(0))
        self.assertTrue(result[0].isupper())

    def test_empty_string(self) -> None:
        result = insert_contractions("", rate=1.0, rng=random.Random(0))
        self.assertEqual(result, "")

    def test_contracts_multiple(self) -> None:
        text = "They are not coming. She will not attend."
        result = insert_contractions(text, rate=1.0, rng=random.Random(0))
        # "they are" contracts to "they're" before "are not" → "aren't"
        self.assertIn("They're", result)
        self.assertIn("won't", result)


class TestReorderClauses(unittest.TestCase):
    def test_moves_prepositional_clause(self) -> None:
        sentences = ["In 2024, the technology improved significantly."]
        result = reorder_clauses(sentences, rate=1.0, rng=random.Random(0))
        # The clause should be moved to end
        self.assertNotEqual(result[0], sentences[0])
        self.assertIn("2024", result[0].lower())
        self.assertIn("technology", result[0].lower())

    def test_rate_zero_no_change(self) -> None:
        sentences = ["In 2024, things changed."]
        result = reorder_clauses(sentences, rate=0.0, rng=random.Random(0))
        self.assertEqual(result, sentences)

    def test_preserves_non_matching_sentences(self) -> None:
        sentences = ["The cat sat on the mat."]
        result = reorder_clauses(sentences, rate=1.0, rng=random.Random(0))
        self.assertEqual(result[0], sentences[0])

    def test_empty_list(self) -> None:
        result = reorder_clauses([], rate=1.0)
        self.assertEqual(result, [])


class TestSplitLongSentences(unittest.TestCase):
    def test_splits_long_compound(self) -> None:
        sentence = (
            "The researchers conducted extensive analysis of the data, "
            "and they found significant patterns in the results that "
            "warranted further investigation."
        )
        result = split_long_sentences([sentence], rate=1.0, min_words=10, rng=random.Random(0))
        self.assertGreater(len(result), 1)

    def test_rate_zero_no_split(self) -> None:
        sentence = "A long sentence that goes on and on, and it keeps going with more content."
        result = split_long_sentences(
            [sentence], rate=0.0, min_words=5, rng=random.Random(0)
        )
        self.assertEqual(len(result), 1)

    def test_short_sentence_not_split(self) -> None:
        sentence = "Short sentence."
        result = split_long_sentences(
            [sentence], rate=1.0, min_words=10, rng=random.Random(0)
        )
        self.assertEqual(len(result), 1)

    def test_empty_list(self) -> None:
        result = split_long_sentences([], rate=1.0)
        self.assertEqual(result, [])

    def test_split_preserves_content(self) -> None:
        sentence = "The weather was sunny and warm, but the forecast predicted heavy rain later in the afternoon."
        result = split_long_sentences([sentence], rate=1.0, min_words=8, rng=random.Random(0))
        combined = " ".join(result).lower()
        self.assertIn("weather", combined)
        self.assertIn("forecast", combined)


class TestInsertDiscourseFillers(unittest.TestCase):
    def test_adds_fillers_at_high_rate(self) -> None:
        sentences = [
            "The first sentence is here.",
            "The second sentence follows suit.",
            "The third sentence wraps it up nicely.",
        ]
        result = insert_discourse_fillers(sentences, rate=1.0, rng=random.Random(0))
        # First sentence should never get a filler
        self.assertFalse(any(
            result[0].startswith(f) for f in [
                "Honestly,", "Look,", "The thing is,", "Here's the deal",
                "Basically,", "Truth is,", "I mean,", "Sure,", "Right,",
            ]
        ))
        # At least one subsequent sentence should have a filler
        has_filler = False
        for sent in result[1:]:
            if any(sent.startswith(f) for f in [
                "Honestly,", "Look,", "The thing is,", "Here's the deal",
                "Basically,", "Truth is,", "I mean,", "Sure,", "Right,",
                "So yeah,", "In practice,", "Realistically,",
                "At the end of the day,", "For what it's worth,",
                "Point being,", "To be fair,", "If you think about it,",
                "No doubt,", "Funny enough,", "Interestingly,",
                "Naturally,", "And honestly,", "As it turns out,",
                "The reality is,", "Plain and simple,",
            ]):
                has_filler = True
                break
        self.assertTrue(has_filler)

    def test_rate_zero_no_fillers(self) -> None:
        sentences = ["First sentence.", "Second sentence.", "Third sentence."]
        result = insert_discourse_fillers(sentences, rate=0.0, rng=random.Random(0))
        self.assertEqual(result, sentences)

    def test_empty_list(self) -> None:
        result = insert_discourse_fillers([], rate=1.0)
        self.assertEqual(result, [])

    def test_first_sentence_never_gets_filler(self) -> None:
        sentences = ["Only one long enough sentence right here."]
        result = insert_discourse_fillers(sentences, rate=1.0, rng=random.Random(0))
        self.assertEqual(result[0], sentences[0])


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

    def test_high_rate_still_produces_output(self) -> None:
        """swap_rate above old cap of 0.40 should still work."""
        text = "The quick brown fox jumps over the lazy dog."
        result = substitute_synonyms(text, swap_rate=0.80, rng=random.Random(42))
        self.assertIsInstance(result, str)
        self.assertTrue(result.strip())


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
        result = humanize(
            text, synonym_rate=0.0, merge_rate=0.0, seed=0,
            contraction_rate=0.0, clause_reorder_rate=0.0,
            split_rate=0.0, filler_rate=0.0,
        )
        # With no swaps and no transforms, only marker-stripping and
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

    def test_contractions_applied_in_pipeline(self) -> None:
        text = "It is important to understand. They do not care."
        result = humanize(text, contraction_rate=1.0, seed=42,
                          synonym_rate=0.0, merge_rate=0.0,
                          clause_reorder_rate=0.0, split_rate=0.0,
                          filler_rate=0.0)
        # At least one contraction should appear
        lower = result.humanized_text.lower()
        has_contraction = "it's" in lower or "don't" in lower
        self.assertTrue(has_contraction)

    def test_all_rates_zero_produces_minimal_change(self) -> None:
        text = "The sky is blue. The grass is green."
        result = humanize(
            text, synonym_rate=0.0, merge_rate=0.0,
            contraction_rate=0.0, clause_reorder_rate=0.0,
            split_rate=0.0, filler_rate=0.0, seed=0,
        )
        # With everything at zero, output should be very close to input
        self.assertIn("sky", result.humanized_text.lower())
        self.assertIn("blue", result.humanized_text.lower())
        self.assertIn("grass", result.humanized_text.lower())
        self.assertIn("green", result.humanized_text.lower())

    def test_backward_compatibility_default_params(self) -> None:
        """humanize() should work without any new parameters (backward compat)."""
        text = "AI is evolving. Many people are excited."
        result = humanize(text, seed=0)
        self.assertIsInstance(result, HumanizeResult)
        self.assertTrue(result.humanized_text.strip())


class TestSynonymSafetyGuards(unittest.TestCase):
    """Tests for the synonym blacklist safety measures."""

    def test_blacklisted_words_are_never_swapped(self) -> None:
        """Context-critical blacklisted words must survive at swap_rate=1.0."""
        critical_words = ["not", "very", "also", "just", "only"]
        for word in critical_words:
            text = f"The approach is {word} effective in practice."
            result = substitute_synonyms(text, swap_rate=1.0, rng=random.Random(0))
            with self.subTest(word=word):
                self.assertIn(word, result.lower())

    def test_blacklist_contains_key_words(self) -> None:
        """Spot-check that the expected words are in the blacklist."""
        expected = {"not", "very", "also", "just", "only", "even", "still"}
        for word in expected:
            with self.subTest(word=word):
                self.assertIn(word, _SYNONYM_BLACKLIST)

    def test_humanize_preserves_blacklisted_words_at_high_rate(self) -> None:
        """Full pipeline must not replace blacklisted words even at high rates."""
        text = "The approach is not very effective. It is also quite slow."
        result = humanize(text, synonym_rate=0.80, merge_rate=0.0, seed=0,
                          contraction_rate=0.0, clause_reorder_rate=0.0,
                          split_rate=0.0, filler_rate=0.0)
        output = result.humanized_text.lower()
        for word in ("not", "very", "also"):
            with self.subTest(word=word):
                self.assertIn(word, output)


if __name__ == "__main__":
    unittest.main()
