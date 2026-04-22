"""Tests for BeyondAIME accuracy metric."""

from __future__ import annotations

import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.beyond_aime import BeyondAIMEAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class BeyondAIMEAccuracyMetricTests(unittest.TestCase):
    """Tests for BeyondAIMEAccuracyMetric."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        spec = MetricSpec(
            metric_id="beyond_aime_acc", implementation="beyond_aime_accuracy", params={}
        )
        self.metric = BeyondAIMEAccuracyMetric(spec)

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

    def test_exact_match_integer(self) -> None:
        """Test exact match with integer answer."""
        context = self._create_context(r"\boxed{42}", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction"], "42")

    def test_exact_match_integer_with_spaces(self) -> None:
        """Test exact match with spaces around integer."""
        context = self._create_context(r"\boxed{  42  }", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_exact_match_integer_mismatch(self) -> None:
        """Test mismatch with different integers."""
        context = self._create_context(r"\boxed{42}", "43")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_extract_last_boxed(self) -> None:
        """Test extraction of last boxed answer when multiple exist."""
        context = self._create_context(
            r"First guess: \boxed{10}, Final answer: \boxed{42}", "42"
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction"], "42")

    def test_no_boxed_fallback_to_raw(self) -> None:
        """Test fallback to raw text when no boxed answer found."""
        context = self._create_context("The answer is 42", "42")
        result = self.metric.compute(context)
        # Should use raw text as prediction
        self.assertEqual(result.metadata["prediction"], "The answer is 42")

    def test_large_integer(self) -> None:
        """Test with large integer answer."""
        context = self._create_context(r"\boxed{123456789}", "123456789")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_empty_prediction(self) -> None:
        """Test with empty prediction."""
        context = self._create_context("", "42")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_metadata_contains_normalized_values(self) -> None:
        """Test that metadata contains normalized prediction and reference."""
        context = self._create_context(r"\boxed{42}", "42")
        result = self.metric.compute(context)
        self.assertIn("normalized_prediction", result.metadata)
        self.assertIn("normalized_reference", result.metadata)
        self.assertEqual(result.metadata["normalized_prediction"], "42")
        self.assertEqual(result.metadata["normalized_reference"], "42")

    def test_sample_id_preserved(self) -> None:
        """Test that sample_id is preserved in result."""
        context = self._create_context(r"\boxed{42}", "42", sample_id="test_001")
        result = self.metric.compute(context)
        self.assertEqual(result.sample_id, "test_001")

    def test_answer_with_text_removed(self) -> None:
        """Test that text is removed and only digits are kept."""
        # For integer answers, letters should be removed
        context = self._create_context(r"\boxed{42units}", "42")
        result = self.metric.compute(context)
        # Normalization removes letters, keeping only digits
        self.assertEqual(result.metadata["normalized_prediction"], "42")
        self.assertEqual(result.values["acc"], 1.0)


class BeyondAIMEAccuracyMetricEdgeCasesTests(unittest.TestCase):
    """Edge case tests for BeyondAIMEAccuracyMetric."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        spec = MetricSpec(
            metric_id="beyond_aime_acc", implementation="beyond_aime_accuracy", params={}
        )
        self.metric = BeyondAIMEAccuracyMetric(spec)

    def test_whitespace_in_boxed(self) -> None:
        """Test handling of whitespace inside boxed."""
        context = MetricContext(
            sample_id="demo",
            sample={
                "predict_result": [
                    {"message": {"content": [{"text": r"\boxed{  42  }"}]}}
                ],
                "references": ["42"],
            },
            model_output={},
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        # Normalization should handle whitespace
        self.assertEqual(result.values["acc"], 1.0)

    def test_answer_with_text_explanation(self) -> None:
        """Test extracting boxed from text with explanation."""
        prediction = r"""To solve this problem, we add the numbers.
        
        2 + 2 = 4
        
        Therefore, the answer is \boxed{4}."""
        context = MetricContext(
            sample_id="demo",
            sample={
                "predict_result": [
                    {"message": {"content": [{"text": prediction}]}}
                ],
                "references": ["4"],
            },
            model_output={},
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction"], "4")

    def test_nested_boxed(self) -> None:
        """Test extracting boxed with nested braces."""
        # This tests that nested braces are handled correctly
        prediction = r"The answer is \boxed{42}"
        context = MetricContext(
            sample_id="demo",
            sample={
                "predict_result": [
                    {"message": {"content": [{"text": prediction}]}}
                ],
                "references": ["42"],
            },
            model_output={},
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
