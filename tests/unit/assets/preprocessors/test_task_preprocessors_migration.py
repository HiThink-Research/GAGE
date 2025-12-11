import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor
from gage_eval.assets.datasets.preprocessors.piqa_preprocessor import PiqaStructOnlyPreprocessor
from gage_eval.assets.datasets.preprocessors.docvqa_preprocessor import DocVQAPreprocessor
from gage_eval.assets.datasets.preprocessors.mmmu_preprocessor import MMMUMultimodalPreprocessor


class TaskPreprocessorMigrationTests(unittest.TestCase):
    def test_multi_choice_flags(self):
        pre = MultiChoicePreprocessor()
        sample = {"question": "Q?", "choices": ["A1", "B2"], "answer": "B"}
        out = pre.transform(sample)
        self.assertEqual(out["prompt"], "Q?\n\nA. A1\nB. B2\n\n请仅输出正确选项对应的大写字母，例如 'A'。")
        self.assertEqual(sample["cache_suffix"], "-converted")
        self.assertEqual(sample["metadata"]["correct_choice"], "B")
        self.assertEqual(sample["choices"][1]["message"]["content"][0]["text"], "B2")

    def test_piqa_struct_only(self):
        pre = PiqaStructOnlyPreprocessor()
        sample = {"question": "Q?", "choices": ["opt1", "opt2"], "answer": "B"}
        pre.transform(sample)
        self.assertEqual(sample.get("inputs"), {})
        self.assertEqual(sample.get("messages"), [])
        self.assertNotIn("prompt", sample)
        self.assertEqual(sample["metadata"]["correct_choice"], "B")

    def test_docvqa_image_merge(self):
        pre = DocVQAPreprocessor()
        sample = {
            "question": "what?",
            "choices": [{"message": {"content": [{"type": "text", "text": "ans"}]}}],
            "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "img.png"}}]}],
        }
        out = pre.transform(sample, content_root="/root")
        self.assertEqual(out["prompt"], "what?")
        self.assertEqual(sample["metadata"]["content_root"], "/root")
        self.assertEqual(sample["metadata"]["answers"], ["ans"])
        self.assertEqual(sample["messages"][0]["content"][1]["image_url"]["url"], "/root/img.png")

    def test_mmmu_multimodal(self):
        pre = MMMUMultimodalPreprocessor()
        sample = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "u.png"}}]}
            ]
        }
        out = pre.transform(sample)
        self.assertIn("multi_modal_data", out)
        self.assertEqual(out["multi_modal_data"]["image"][0], "u.png")
        self.assertEqual(sample["cache_suffix"], "-converted")
        self.assertEqual(sample["chat_template_mode"], "preprocess")


if __name__ == "__main__":
    unittest.main()
