import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.metrics.utils import extract_field
from gage_eval.metrics.base import MetricContext


class FieldPathTests(unittest.TestCase):
    def setUp(self):
        self.sample = {
            "metadata": {"correct_choice": "B", "option_map": {"A": "x", "B": "y"}},
            "label": "LBL",
            "choices": [{"message": {"content": [{"type": "text", "text": "ans"}]}}],
        }
        self.model_output = {"answer": "B"}
        self.judge_output = {"answer": "A"}
        self.context = MetricContext(
            sample=self.sample,
            model_output=self.model_output,
            judge_output=self.judge_output,
            sample_id="sid",
            args={},
            trace=None,
        )

    def test_multi_choice_paths(self):
        expected = extract_field(self.context, "sample.metadata.correct_choice")
        prediction = extract_field(self.context, "model_output.answer")
        option_map = extract_field(self.context, "sample.metadata.option_map")
        self.assertEqual(expected, "B")
        self.assertEqual(prediction, "B")
        self.assertEqual(option_map["B"], "y")

    def test_predict_result_path(self):
        self.sample["predict_result"] = [
            {"message": {"content": [{"type": "text", "text": "B"}]}},
        ]
        ctx = MetricContext(
            sample=self.sample,
            model_output=self.model_output,
            judge_output=self.judge_output,
            sample_id="sid",
            args={},
            trace=None,
        )
        prediction = extract_field(ctx, "sample.predict_result.0.message.content.0.text")
        self.assertEqual(prediction, "B")

    def test_mmnu_label_default(self):
        label = extract_field(self.context, "sample.choices.0.message.content.0.text")
        self.assertEqual(label, "ans")


if __name__ == "__main__":
    unittest.main()
