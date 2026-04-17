"""Tests for GSM8K accuracy metric."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.gsm8k import GSM8KAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class GSM8KAccuracyMetricTests(unittest.TestCase):
    """Tests for GSM8KAccuracyMetric."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        spec = MetricSpec(
            metric_id="gsm8k_acc", implementation="gsm8k_accuracy", params={}
        )
        self.metric = GSM8KAccuracyMetric(spec)

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

    def test_exact_match_answer_format(self) -> None:
        """Test exact match with ANSWER format."""
        context = self._create_context("The answer is ANSWER: 42", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction"], "42")

    def test_exact_match_boxed_format(self) -> None:
        """Test exact match with \\boxed{} format."""
        context = self._create_context(r"The answer is \boxed{42}", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction"], "42")

    def test_boxed_nested_braces(self) -> None:
        """Test \\boxed{} with nested braces selects last match."""
        context = self._create_context(
            r"First: \boxed{10}, Final: \boxed{42}", "42"
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_boxed_with_whitespace(self) -> None:
        """Test \\boxed{} with surrounding whitespace."""
        context = self._create_context(
            r"The answer is \boxed{  42  }", "42"
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_answer_format_priority_over_boxed(self) -> None:
        """ANSWER: takes priority over \\boxed{} when both present."""
        context = self._create_context(
            r"Guess: \boxed{10}, Final answer: ANSWER: 42", "42"
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_mismatch(self) -> None:
        """Test mismatch with different numbers."""
        context = self._create_context("The answer is ANSWER: 42", "43")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_comma_removal_reference(self) -> None:
        """Test that commas are removed from reference before comparison."""
        context = self._create_context("ANSWER: 1234", "1,234")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["references"], "1234")

    def test_no_answer_marker_fallback_to_number(self) -> None:
        """Test prediction without ANSWER or boxed falls back to numeric extraction."""
        context = self._create_context("The answer is 42", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_negative_number(self) -> None:
        """Test with negative answer."""
        context = self._create_context("ANSWER: -5", "-5")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_negative_number_boxed(self) -> None:
        """Test with negative answer in boxed format."""
        context = self._create_context(r"\boxed{-5}", "-5")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_sample_id_preserved(self) -> None:
        """Test that sample_id is preserved in result."""
        context = self._create_context("ANSWER: 42", "42", sample_id="test_001")
        result = self.metric.compute(context)
        self.assertEqual(result.sample_id, "test_001")

    def test_multiline_prediction(self) -> None:
        """Test extracting last ANSWER from multi-line response."""
        prediction = """Step 1: add 1 and 1
Step 2: get 2
ANSWER: 2"""
        context = self._create_context(prediction, "2")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_multiple_answer_lines(self) -> None:
        """Test regex selects last ANSWER occurrence."""
        context = self._create_context("ANSWER: 33\nANSWER: 34", "34")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_multiple_answer_lines_mismatch(self) -> None:
        """Test regex selects last ANSWER occurrence and fails when mismatched."""
        context = self._create_context("ANSWER: 33\nANSWER: 34", "33")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)


if __name__ == "__main__":
    unittest.main()
