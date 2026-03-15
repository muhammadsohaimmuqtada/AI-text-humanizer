"""Integration tests for the AI Text Humanizer CLI and API.

Replaces the old deepfake-detection evaluation pipeline tests with tests
that exercise the new humanization endpoints end-to-end.
"""

from __future__ import annotations

import json
import sys
import unittest

from aip.cli import main as cli_main


class CliHumanizeTests(unittest.TestCase):
    _SAMPLE = (
        "Furthermore, it is important to note that AI is rapidly evolving. "
        "Many businesses use it to automate tasks. "
        "In conclusion, the future looks bright."
    )

    def _run_cli(self, *args: str) -> dict:
        """Run the CLI with the given args and parse the JSON output."""
        import io
        import contextlib

        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = cli_main(list(args))
        self.assertEqual(rc, 0, f"CLI returned non-zero: {buf.getvalue()}")
        return json.loads(buf.getvalue())

    def test_humanize_returns_json_with_required_keys(self) -> None:
        out = self._run_cli("humanize", "--text", self._SAMPLE)
        for key in ("humanized_text", "original_word_count", "humanized_word_count",
                    "markers_removed", "sentences_merged"):
            self.assertIn(key, out, f"missing key: {key}")

    def test_humanize_removes_markers(self) -> None:
        out = self._run_cli("humanize", "--text", self._SAMPLE)
        self.assertGreater(out["markers_removed"], 0)
        self.assertNotIn("in conclusion", out["humanized_text"].lower())
        self.assertNotIn("furthermore", out["humanized_text"].lower())

    def test_humanize_word_counts_are_positive(self) -> None:
        out = self._run_cli("humanize", "--text", self._SAMPLE)
        self.assertGreater(out["original_word_count"], 0)
        self.assertGreater(out["humanized_word_count"], 0)

    def test_humanize_deterministic_with_seed(self) -> None:
        out1 = self._run_cli("humanize", "--text", self._SAMPLE, "--seed", "42")
        out2 = self._run_cli("humanize", "--text", self._SAMPLE, "--seed", "42")
        self.assertEqual(out1["humanized_text"], out2["humanized_text"])

    def test_humanize_with_merge_rate_zero(self) -> None:
        out = self._run_cli("humanize", "--text", self._SAMPLE, "--merge-rate", "0.0")
        self.assertEqual(out["sentences_merged"], 0)
        self.assertTrue(out["humanized_text"].strip())

    def test_doctor_returns_runtime_capabilities(self) -> None:
        out = self._run_cli("doctor")
        self.assertIn("runtime_capabilities", out)
        caps = out["runtime_capabilities"]
        self.assertIn("nltk", caps)
        self.assertIn("wordnet", caps)
        self.assertIn("punkt", caps)


class ApiRouteTests(unittest.TestCase):
    def setUp(self) -> None:
        from aip.api import create_app

        try:
            from starlette.testclient import TestClient
        except ImportError:  # pragma: no cover
            self.skipTest("starlette.testclient not available")

        self.client = TestClient(create_app())

    def test_healthz(self) -> None:
        r = self.client.get("/healthz")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["status"], "ok")

    def test_readyz_returns_status(self) -> None:
        r = self.client.get("/readyz")
        self.assertEqual(r.status_code, 200)
        self.assertIn("status", r.json())
        self.assertIn("runtime_capabilities", r.json())

    def test_humanize_endpoint_success(self) -> None:
        r = self.client.post(
            "/humanize",
            json={
                "text": "Furthermore, AI is rapidly evolving. It is important to note that many businesses use it.",
                "seed": 42,
            },
        )
        self.assertEqual(r.status_code, 200)
        data = r.json()
        self.assertIn("humanized_text", data)
        self.assertIn("original_word_count", data)
        self.assertIn("humanized_word_count", data)
        self.assertIn("markers_removed", data)
        self.assertIn("sentences_merged", data)
        self.assertGreater(data["markers_removed"], 0)

    def test_humanize_empty_text(self) -> None:
        r = self.client.post("/humanize", json={"text": ""})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["original_word_count"], 0)

    def test_policies_endpoint(self) -> None:
        r = self.client.get("/policies")
        self.assertEqual(r.status_code, 200)
        self.assertIn("max_text_chars", r.json())

    def test_metrics_endpoint(self) -> None:
        r = self.client.get("/metrics")
        self.assertEqual(r.status_code, 200)


if __name__ == "__main__":
    unittest.main()
