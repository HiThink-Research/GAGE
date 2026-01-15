import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.mmlu_pro.acc import MMLUProAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec

class MMLUProAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="mmlu_pro_acc", implementation="mmlu_pro_acc", params={})
        self.metric = MMLUProAccuracyMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "the answer is (B)"}]}}],
                    "references": ["B"],
                    "metadata": {
                    }
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_acc_choice(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "the answer is D"}]}}],
                    "references": ["D"],
                    "metadata": {
                        "answer_type": "multipleChoice"
                    }
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
