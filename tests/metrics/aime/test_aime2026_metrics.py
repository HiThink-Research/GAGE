"""Tests for AIME 2026 accuracy metric."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.aime import AIME2026AccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class AIME2026AccuracyMetricTests(unittest.TestCase):
    """Tests for AIME2026AccuracyMetric."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        spec = MetricSpec(
            metric_id="aime2026_acc", implementation="aime2026_accuracy", params={}
        )
        self.metric = AIME2026AccuracyMetric(spec)

    def _create_context(
        self, prediction_text: str, reference: str, sample_id: str = "demo"
    ) -> MetricContext:
        """Helper to create a MetricContext for testing."""
        return MetricContext(
            sample_id=sample_id,
            sample={
                "predict_result": [
                    {"message": {"content": [{"text": prediction_text}]}}
                ],
                "references": [reference],
            },
            model_output={},
            judge_output={},
            args={},
            trace=None,
        )

    def test_exact_match(self) -> None:
        """Test exact match with integer answer."""
        context = self._create_context("ANSWER: 33", "33")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_mismatch(self) -> None:
        """Test mismatch with different integers."""
        context = self._create_context("ANSWER: 33", "34")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_regex_multiple_lines(self) -> None:
        """Test regex selects last occurrence."""
        context = self._create_context("ANSWER: 33\nANSWER: 34", "34")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_regex_wrong_multiple_lines(self) -> None:
        """Test regex selects last occurrence and fails when mismatched."""
        context = self._create_context("ANSWER: 33\nANSWER: 34", "33")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_sample_id_preserved(self) -> None:
        """Test that sample_id is preserved in result."""
        context = self._create_context("ANSWER: 42", "42", sample_id="test_001")
        result = self.metric.compute(context)
        self.assertEqual(result.sample_id, "test_001")

    def test_metadata_contains_prediction(self) -> None:
        """Test that metadata contains prediction and references."""
        context = self._create_context("ANSWER: 42", "42")
        result = self.metric.compute(context)
        self.assertIn("prediction", result.metadata)
        self.assertIn("references", result.metadata)
        self.assertEqual(result.metadata["prediction"], "42")
        self.assertEqual(result.metadata["references"], "42")


if __name__ == "__main__":
    unittest.main()
