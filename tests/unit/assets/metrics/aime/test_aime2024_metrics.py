import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.aime import AIME2024AccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec

class AIME2024AccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="aime_acc", implementation="aime2024_accuracy", params={})
        self.metric = AIME2024AccuracyMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "ANSWER: 33"}]}}],
                    "references": ["33"]
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_regex(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "ANSWER: 33\nANSWER: 34"}]}}],
                    "references": ["33"]
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_regex_select(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "ANSWER: 33\nANSWER: 34"}]}}],
                    "references": ["34"]
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
   
if __name__ == "__main__":
    unittest.main()
