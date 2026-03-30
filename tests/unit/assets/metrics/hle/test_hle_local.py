import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.hle.hle_local import HLEAccuracyLocalMetric
from gage_eval.config.pipeline_config import MetricSpec

class AIME2024AccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="hle_acc", implementation="hle_accuracy_local", params={})
        self.metric = HLEAccuracyLocalMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": "ANSWER: 33 \nConfidence: 100%"}]}}],
                    "references": ["33"],
                    "metadata": {
                        "answer_type": "exactMatch"
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
            sample={"predict_result": [{"message": {"content": [{"text": "ANSWER: the correct one is D \nConfidence: 100%"}]}}],
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
