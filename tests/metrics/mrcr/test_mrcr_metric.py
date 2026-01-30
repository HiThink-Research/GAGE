import sys
import unittest

ROOT = __file__.rsplit("/tests/", 1)[0] + "/src"
if ROOT not in sys.path:
    sys.path.append(ROOT)

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.mrcr.mrcr_grade import MRCRGradeMetric
from gage_eval.config.pipeline_config import MetricSpec

class MRCRGradeMetricTests(unittest.TestCase):
    def setUp(self) -> None:
        spec = MetricSpec(metric_id="mmlu_pro_acc", implementation="mmlu_pro_acc", params={})
        self.metric = MRCRGradeMetric(spec)

    def test_acc(self):
        metadata = {
            "random_string_to_prepend": "mWEa9DrPT3",
            "n_needles": 2,
            "desired_msg_index": 721,
            "date_added": "04-12-2025"
        }        
        context = MetricContext(
            sample_id="demo",
            sample = {
                "prompt":  '[{"role": "user", "content": "are you ok?"}]',
                "answer": "mWEa9DrPT3**Verse 1** \nIn a world so vast",
                "random_string_to_prepend": "mWEa9DrPT3",
                "n_needles": 2,
                "desired_msg_index": 721,
                "total_messages": 772,
                "n_chars": 708925,
                "date_added": "04-12-2025",
                "predict_result": [{"message": {"content": [{"text": "mWEa9DrPT3**Verse 1** \nIn a world so vast"}]}}],
                "references": ["mWEa9DrPT3**Verse 1** \nIn a world so vast"],
                "metadata": metadata                
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["grade"], 1.0)
        
    def test_acc2(self):
        metadata = {
            "random_string_to_prepend": "mWEa9DrPT3",
            "n_needles": 2,
            "desired_msg_index": 721,
            "date_added": "04-12-2025"
        }        
        context = MetricContext(
            sample_id="demo",
            sample = {
                "prompt":  '[{"role": "user", "content": "are you ok?"}]',
                "answer": "mWEa9DrPT3**Verse 1** \nIn a world so vast",
                "random_string_to_prepend": "mWEa9DrPT3",
                "n_needles": 2,
                "desired_msg_index": 721,
                "total_messages": 772,
                "n_chars": 708925,
                "date_added": "04-12-2025",
                "predict_result": [{"message": {"content": [{"text": "**Verse 1** \nIn a world so vast"}]}}],
                "references": ["mWEa9DrPT3**Verse 1** \nIn a world so vast"],
                "metadata": metadata                
            },
            model_output={
            },
            judge_output={},
            args={},
            trace=None,
        )
        result = self.metric.compute(context)
        self.assertEqual(result.values["grade"], 0.0)

   
if __name__ == "__main__":
    unittest.main()
