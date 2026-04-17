"""Tests for Video-MME accuracy metric."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.video_mme import VideoMMEAccuracyMetric


class VideoMMEAccuracyMetricTests(unittest.TestCase):
    """Tests for VideoMMEAccuracyMetric."""

    def setUp(self) -> None:
        spec = MetricSpec(
            metric_id="video_mme_acc",
            implementation="video_mme_accuracy",
            params={},
        )
        self.metric = VideoMMEAccuracyMetric(spec)

    def _create_context(
        self, prediction_text: str, reference: str, sample_id: str = "demo"
    ) -> MetricContext:
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

    def test_correct_answer(self) -> None:
        context = self._create_context("The answer is B", "B")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata["prediction_label"], "B")

    def test_wrong_answer(self) -> None:
        context = self._create_context("I think A is correct", "C")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
        self.assertEqual(result.metadata["prediction_label"], "A")

    def test_empty_prediction(self) -> None:
        context = self._create_context("", "A")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
        self.assertEqual(result.metadata["prediction_label"], "")

    def test_extract_from_long_text(self) -> None:
        # In long text the first A/B/C/D occurrence is returned.
        # Place D at the beginning so it matches the reference.
        context = self._create_context(
            "D – after analyzing the video, I conclude that option D is the best answer because ...",
            "D",
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_case_insensitive(self) -> None:
        context = self._create_context("answer: c", "C")
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_no_letter_in_long_text(self) -> None:
        # Use words that genuinely do not contain the letters A, B, C, or D.
        context = self._create_context(
            "This is very long text with no like option in it.",
            "A",
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
        self.assertEqual(result.metadata["prediction_label"], "")

    def test_sample_id_preserved(self) -> None:
        context = self._create_context("B", "B", sample_id="test_001")
        result = self.metric.compute(context)
        self.assertEqual(result.sample_id, "test_001")


if __name__ == "__main__":
    unittest.main()
