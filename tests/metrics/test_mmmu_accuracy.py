import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.mmmu import MMMUAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class MMMUAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="mmmu_acc", implementation="mmmu_accuracy", params={})
        self.metric = MMMUAccuracyMetric(spec)

    def test_rejects_letter_in_explanation(self):
        # target is C, model explanation mentions "Option C..." but final answer boxed D.
        context = MetricContext(
            sample_id="demo",
            sample={"choices": [{"message": {"content": [{"text": "C"}]}}]},
            model_output={
                "answer": 'Option C is too broad. <answer> D </answer>',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_accepts_boxed_letter(self):
        context = MetricContext(
            sample_id="demo",
            sample={"choices": [{"message": {"content": [{"text": "B"}]}}]},
            model_output={
                "answer": "reasoning... \\boxed{B}",
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_accepts_answer_tag_letter(self):
        context = MetricContext(
            sample_id="demo",
            sample={"choices": [{"message": {"content": [{"text": "A"}]}}]},
            model_output={
                "answer": "<answer> A </answer>",
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
