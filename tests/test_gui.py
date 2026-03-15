"""Unit tests for the GUI bypass-strength feature.

These tests exercise the ``_BYPASS_PRESETS`` constant and the multi-pass
worker logic in isolation (no display required).
"""

from __future__ import annotations

import sys
import unittest

# ---------------------------------------------------------------------------
# Conditional import: _BYPASS_PRESETS lives in aip.gui which imports tkinter.
# On headless CI runners tkinter may not be installed.  We skip the preset
# tests gracefully in that case and rely on the multi-pass logic tests instead.
# ---------------------------------------------------------------------------
try:
    from aip.gui import _BYPASS_PRESETS, _DEFAULT_STRENGTH

    _GUI_AVAILABLE = True
except ModuleNotFoundError:
    _GUI_AVAILABLE = False
    _BYPASS_PRESETS = {}  # type: ignore[assignment]
    _DEFAULT_STRENGTH = "Aggressive"

_SKIP_GUI = unittest.skipUnless(_GUI_AVAILABLE, "tkinter not available in this environment")


@_SKIP_GUI
class TestBypassPresets(unittest.TestCase):
    """Validate the structure and semantics of _BYPASS_PRESETS."""

    def test_presets_contain_expected_strength_levels(self) -> None:
        for expected in ("Light", "Medium", "Aggressive", "Custom"):
            self.assertIn(expected, _BYPASS_PRESETS)

    def test_non_custom_presets_have_required_keys(self) -> None:
        for name, preset in _BYPASS_PRESETS.items():
            if preset is None:
                continue  # Custom is explicitly None
            with self.subTest(strength=name):
                self.assertIn("passes", preset)
                self.assertIn("synonym_rate", preset)
                self.assertIn("merge_rate", preset)

    def test_custom_preset_is_none(self) -> None:
        self.assertIsNone(_BYPASS_PRESETS["Custom"])

    def test_passes_increase_with_strength(self) -> None:
        light = _BYPASS_PRESETS["Light"]["passes"]
        medium = _BYPASS_PRESETS["Medium"]["passes"]
        aggressive = _BYPASS_PRESETS["Aggressive"]["passes"]
        self.assertLess(light, medium)
        self.assertLess(medium, aggressive)

    def test_synonym_rate_increases_with_strength(self) -> None:
        self.assertLess(
            _BYPASS_PRESETS["Light"]["synonym_rate"],
            _BYPASS_PRESETS["Medium"]["synonym_rate"],
        )
        self.assertLess(
            _BYPASS_PRESETS["Medium"]["synonym_rate"],
            _BYPASS_PRESETS["Aggressive"]["synonym_rate"],
        )

    def test_merge_rate_increases_with_strength(self) -> None:
        self.assertLess(
            _BYPASS_PRESETS["Light"]["merge_rate"],
            _BYPASS_PRESETS["Medium"]["merge_rate"],
        )
        self.assertLess(
            _BYPASS_PRESETS["Medium"]["merge_rate"],
            _BYPASS_PRESETS["Aggressive"]["merge_rate"],
        )

    def test_rates_are_within_valid_range(self) -> None:
        for name, preset in _BYPASS_PRESETS.items():
            if preset is None:
                continue
            with self.subTest(strength=name):
                self.assertGreaterEqual(preset["synonym_rate"], 0.0)
                self.assertLessEqual(preset["synonym_rate"], 1.0)
                self.assertGreaterEqual(preset["merge_rate"], 0.0)
                self.assertLessEqual(preset["merge_rate"], 1.0)

    def test_default_strength_is_a_valid_preset(self) -> None:
        self.assertIn(_DEFAULT_STRENGTH, _BYPASS_PRESETS)

    def test_light_preset_runs_single_pass(self) -> None:
        self.assertEqual(_BYPASS_PRESETS["Light"]["passes"], 1)

    def test_aggressive_preset_runs_three_passes(self) -> None:
        self.assertEqual(_BYPASS_PRESETS["Aggressive"]["passes"], 3)


# ---------------------------------------------------------------------------
# These tests exercise only aip.humanizer (no tkinter dependency).
# ---------------------------------------------------------------------------


class TestMultiPassLogic(unittest.TestCase):
    """Verify that multi-pass humanisation produces progressively different output."""

    # Preset values mirrored from _BYPASS_PRESETS so we can test without tkinter.
    # Keep in sync with aip/gui.py _BYPASS_PRESETS when updating preset values.
    _PRESETS = {
        "Light":      {"passes": 1, "synonym_rate": 0.35, "merge_rate": 0.25},
        "Medium":     {"passes": 2, "synonym_rate": 0.60, "merge_rate": 0.40},
        "Aggressive": {"passes": 3, "synonym_rate": 0.85, "merge_rate": 0.60},
    }

    def _run_passes(self, text: str, passes: int, synonym_rate: float, merge_rate: float) -> str:
        """Helper that mimics the _humanize_worker multi-pass loop."""
        from aip.humanizer import humanize

        current = text
        for _ in range(passes):
            result = humanize(current, synonym_rate=synonym_rate, merge_rate=merge_rate)
            current = result.humanized_text
        return current

    def test_single_pass_produces_non_empty_output(self) -> None:
        text = "Furthermore, AI is rapidly evolving. It is important to note that this changes everything."
        out = self._run_passes(text, passes=1, synonym_rate=0.35, merge_rate=0.25)
        self.assertTrue(out.strip())

    def test_aggressive_three_passes_removes_ai_markers(self) -> None:
        text = "Furthermore, AI is rapidly evolving. In conclusion, things look bright."
        out = self._run_passes(text, passes=3, synonym_rate=0.85, merge_rate=0.60)
        self.assertNotIn("furthermore", out.lower())
        self.assertNotIn("in conclusion", out.lower())

    def test_multi_pass_cumulative_stats(self) -> None:
        """Stats should accumulate correctly across passes."""
        from aip.humanizer import humanize

        text = "Furthermore, AI is evolving fast. In conclusion, it is remarkable."
        total_markers = 0
        total_merges = 0
        current = text
        for _ in range(3):
            result = humanize(current, synonym_rate=0.85, merge_rate=0.60)
            total_markers += result.markers_removed
            total_merges += result.sentences_merged
            current = result.humanized_text

        # First pass should strip the AI markers; total > 0
        self.assertGreater(total_markers, 0)
        self.assertIsInstance(total_merges, int)

    def test_preset_values_are_consistent(self) -> None:
        """Sanity-check the inline preset table used by this test class."""
        presets = self._PRESETS
        self.assertLess(presets["Light"]["passes"], presets["Medium"]["passes"])
        self.assertLess(presets["Medium"]["passes"], presets["Aggressive"]["passes"])
        self.assertLess(presets["Light"]["synonym_rate"], presets["Aggressive"]["synonym_rate"])

    def test_medium_two_passes_produces_output(self) -> None:
        text = "AI systems are becoming incredibly powerful and highly advanced."
        out = self._run_passes(text, **self._PRESETS["Medium"])
        self.assertTrue(out.strip())

    def test_aggressive_three_passes_produces_output(self) -> None:
        text = "AI systems are becoming incredibly powerful and highly advanced. Many researchers study them closely."
        out = self._run_passes(text, **self._PRESETS["Aggressive"])
        self.assertTrue(out.strip())


if __name__ == "__main__":
    unittest.main()
