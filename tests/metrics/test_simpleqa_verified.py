import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.simpleqa_verified import (
    SimpleQAVerifiedAccuracyMetric,
    SimpleQAVerifiedJudgeAccuracyMetric,
)
from gage_eval.config.pipeline_config import MetricSpec


class SimpleQAVerifiedAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="simpleqa_verified_acc", implementation="simpleqa_verified_accuracy", params={})
        self.metric = SimpleQAVerifiedAccuracyMetric(spec)

    def test_exact_match(self):
        """Test exact match for answers."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'Hasnain Masoodi'
                            }
                        ]
                    }
                }],
                'references': ['Hasnain Masoodi'],
                'metadata': {}
            },
            model_output={
                "answer": 'Hasnain Masoodi',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_normalized_match(self):
        """Test normalized match (whitespace differences)."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': '  Hasnain Masoodi  '
                            }
                        ]
                    }
                }],
                'references': ['Hasnain Masoodi'],
                'metadata': {}
            },
            model_output={
                "answer": '  Hasnain Masoodi  ',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        # Should match after normalization
        self.assertEqual(result.values["acc"], 1.0)

    def test_no_match(self):
        """Test no match case."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'John Doe'
                            }
                        ]
                    }
                }],
                'references': ['Hasnain Masoodi'],
                'metadata': {}
            },
            model_output={
                "answer": 'John Doe',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_empty_prediction(self):
        """Test empty prediction."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': ''
                            }
                        ]
                    }
                }],
                'references': ['Hasnain Masoodi'],
                'metadata': {}
            },
            model_output={
                "answer": '',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_multiple_references(self):
        """Test matching against multiple references."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': '2023'
                            }
                        ]
                    }
                }],
                'references': ['2022', '2023', '2024'],
                'metadata': {}
            },
            model_output={
                "answer": '2023',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        # Should match one of the references
        self.assertEqual(result.values["acc"], 1.0)


if __name__ == "__main__":
    unittest.main()


class SimpleQAVerifiedJudgeAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(
            metric_id="simpleqa_verified_judge_acc",
            implementation="simpleqa_verified_judge_accuracy",
            params={},
        )
        self.metric = SimpleQAVerifiedJudgeAccuracyMetric(spec)

    def test_judge_letter_a_correct(self):
        context = MetricContext(
            sample_id="demo",
            sample={"references": ["x"], "metadata": {}},
            model_output={"answer": "x"},
            judge_output={"answer": "A"},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
        self.assertEqual(result.metadata.get("verdict"), "CORRECT")

    def test_judge_letter_b_incorrect(self):
        context = MetricContext(
            sample_id="demo",
            sample={"references": ["x"], "metadata": {}},
            model_output={"answer": "y"},
            judge_output={"answer": "B"},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
        self.assertEqual(result.metadata.get("verdict"), "INCORRECT")

    def test_judge_letter_c_not_attempted(self):
        context = MetricContext(
            sample_id="demo",
            sample={"references": ["x"], "metadata": {}},
            model_output={"answer": "I don't know"},
            judge_output={"answer": "C"},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
        self.assertEqual(result.metadata.get("verdict"), "NOT_ATTEMPTED")
