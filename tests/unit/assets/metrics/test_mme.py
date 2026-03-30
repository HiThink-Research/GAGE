import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.mme import MMEAccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class MMEAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="mme_acc", implementation="mme_accuracy", params={})
        self.metric = MMEAccuracyMetric(spec)

    def test_yes(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Yes'
                            }
                        ]
                    }
                }],
                'references': ['Yes'],
            },
            model_output={
                "answer": 'Yes',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_no(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'No'
                            }
                        ]
                    }
                }],
                'references': ['No'],
            },
            model_output={
                "answer": 'No',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_mismatch(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Yes'
                            }
                        ]
                    }
                }],
                'references': ['No'],
            },
            model_output={
                "answer": 'Yes',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_normalized_match(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'yes'
                            }
                        ]
                    }
                }],
                'references': ['Yes'],
            },
            model_output={
                "answer": 'yes',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

if __name__ == "__main__":
    unittest.main()
