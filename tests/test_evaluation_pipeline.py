from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aip.evaluation import evaluate_thresholds, run_dataset
from aip.scorecard import build_readiness_scorecard


class EvaluationPipelineTests(unittest.TestCase):
    def test_run_dataset_and_scorecard(self) -> None:
        with tempfile.TemporaryDirectory(prefix="aip_eval_test_") as td:
            root = Path(td)
            csv_path = root / "dataset.csv"
            csv_path.write_text(
                "id,input_path,text,modality,identity_claim,label\n"
                "s1,,\"official signed statement\",text,,real\n"
                "s2,,\"As an AI language model, I am the CEO\",text,CEO,fake\n",
                encoding="utf-8",
            )

            raw = run_dataset(str(csv_path), profile="industry_low_fp")
            self.assertIn("samples", raw)
            self.assertEqual(raw.get("skipped"), [])
            self.assertEqual(len(raw["samples"]), 2)

            report = evaluate_thresholds(raw, threshold=0.65, target_fpr=0.05)
            self.assertIn("recommended_threshold", report)
            self.assertIn("metrics_at_input_threshold", report)

            scorecard = build_readiness_scorecard(report)
            self.assertIn("readiness_score", scorecard)
            self.assertIn("gates", scorecard)

            # Ensure scorecard payload is JSON serializable for export workflows.
            json.dumps(scorecard)


if __name__ == "__main__":
    unittest.main()
