import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.mathvista import MathVistaChataccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class MMMUAccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="mathvista_acc", implementation="mathvista_chat_accuracy", params={})
        self.metric = MathVistaChataccuracyMetric(spec)

    def test_float(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': '1.0'
                            }
                        ]
                    }
                }],
                'references': ['1.2'],
                'metadata': {
                    'question_type': 'free_form', 
                    'answer_type': 'float', 
                    'shot_type': 'solution', 
                    'use_caption': True, 'use_ocr': True
                    }
            },
            model_output={
                "answer": 'Option C is too broad. <answer> D </answer>',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_integer(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': '36'
                            }
                        ]
                    }
                }],
                'references': ['36'],
                'metadata': {
                    'question_type': 'free_form', 
                    'answer_type': 'integer', 
                    'shot_type': 'solution', 
                    'use_caption': True, 'use_ocr': True
                    }
            },
            model_output={
                "answer": 'Option C is too broad. <answer> D </answer>',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)

    def test_text(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'I am ok'
                            }
                        ]
                    }
                }],
                'references': ['I am ok'],
                'metadata': {
                    'question_type': 'free_form', 
                    'answer_type': 'text', 
                    'shot_type': 'solution', 
                    'use_caption': True, 'use_ocr': True
                    }
            },
            model_output={
                "answer": 'Option C is too broad. <answer> D </answer>',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 1.0)
    def test_multi_choice(self):
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': 'A'
                            }
                        ]
                    }
                }],
                'references': ['C'],
                'metadata': {
                    'question_type': 'free_form', 
                    'answer_type': 'multi_choice', 
                    'shot_type': 'solution', 
                    'use_caption': True, 'use_ocr': True
                    }
            },
            model_output={
                "answer": 'Option C is too broad. <answer> D </answer>',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)
if __name__ == "__main__":
    unittest.main()
