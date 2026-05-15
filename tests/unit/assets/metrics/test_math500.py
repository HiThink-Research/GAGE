import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.math500 import Math500AccuracyMetric
from gage_eval.config.pipeline_config import MetricSpec


class Math500AccuracyMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="math500_acc", implementation="math500_accuracy", params={})
        self.metric = Math500AccuracyMetric(spec)

    def test_exact_match(self):
        """Test exact match for LaTeX answers."""
        context = MetricContext(
            sample_id="demo",
            sample={
                'predict_result': [{
                    'message': {
                        'role': 'assistant',
                        'content': [
                            {
                                'type': 'text',
                                'text': '\\left( 3, \\frac{\\pi}{2} \\right)'
                            }
                        ]
                    }
                }],
                'references': ['\\left( 3, \\frac{\\pi}{2} \\right)'],
                'metadata': {}
            },
            model_output={
                "answer": '\\left( 3, \\frac{\\pi}{2} \\right)',
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
                                'text': '\\left(3,\\frac{\\pi}{2}\\right)'
                            }
                        ]
                    }
                }],
                'references': ['\\left( 3, \\frac{\\pi}{2} \\right)'],
                'metadata': {}
            },
            model_output={
                "answer": '\\left(3,\\frac{\\pi}{2}\\right)',
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
                                'text': '\\left( 2, \\frac{\\pi}{3} \\right)'
                            }
                        ]
                    }
                }],
                'references': ['\\left( 3, \\frac{\\pi}{2} \\right)'],
                'metadata': {}
            },
            model_output={
                "answer": '\\left( 2, \\frac{\\pi}{3} \\right)',
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["acc"], 0.0)

    def test_extracts_final_answer_instead_of_problem_coordinate(self):
        """Prefer the final answer sentence over earlier coordinates quoted from the prompt."""
        answer = """To convert the point from rectangular coordinates $(x, y) = (0, 3)$ to polar coordinates $(r, \\theta)$:

1. Calculate $r = \\sqrt{0^2 + 3^2} = 3$.
2. Since $\\cos \\theta = 0$ and $\\sin \\theta = 1$, we get $\\theta = \\frac{\\pi}{2}$.

**Conclusion:**
The polar coordinates are $(3, \\frac{\\pi}{2})$.

$(3, \\frac{\\pi}{2})$"""
        context = MetricContext(
            sample_id="demo",
            sample={
                "predict_result": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": [{"type": "text", "text": answer}],
                        }
                    }
                ],
                "references": ["\\left( 3, \\frac{\\pi}{2} \\right)"],
                "metadata": {},
            },
            model_output={"answer": answer},
            judge_output={},
            args={},
            trace=None,
        )

        result = self.metric.compute(context)

        self.assertEqual(result.values["acc"], 1.0)
        self.assertNotEqual(result.metadata["prediction"], "(0, 3)")

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
                'references': ['\\left( 3, \\frac{\\pi}{2} \\right)'],
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


if __name__ == "__main__":
    unittest.main()
