import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.biz_fin_bench_v2.stock_price_predict_short_term import StockPricePredictAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec

class StockPricePredictAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="global_acc", implementation="global_accuracy", params={})
        self.metric = StockPricePredictAccuracyMetric(spec)

    def test_acc(self):
        context = MetricContext(
            sample_id="demo",
            sample={"predict_result": [{"message": {"content": [{"text": 'answer_boxed{50.00}\ninterval_boxed{[49.50,50.50]}'
}]}}],
                    "references": ['50']
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