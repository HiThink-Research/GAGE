import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.biz_fin_bench_v2.aea2 import AEA2AccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec

class AEA2AccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="global_acc", implementation="global_accuracy", params={})
        self.metric = AEA2AccuracyMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": '{"相关内容序号": [10, 9, 11, 16, 20],"type": "text"}'
}]}}],
                    "references": ["9, 10, 11, 16, 20"]
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
