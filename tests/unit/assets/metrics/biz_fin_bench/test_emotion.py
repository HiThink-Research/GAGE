import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.biz_fin_bench_v2.emotion import EvalEmotionAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec

class EvalEmotionAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="global_acc", implementation="global_accuracy", params={})
        self.metric = EvalEmotionAccuracyMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": 'answer_boxed{50.0}\ninterval_boxed{[45,55]}'
}]}}],
                    "references": ["50"]
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
