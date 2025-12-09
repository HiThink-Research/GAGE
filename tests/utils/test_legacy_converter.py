import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.utils.legacy import convert_llmeval_record


class LegacyConverterTests(unittest.TestCase):
    def test_basic_conversion_with_audit_and_prompt(self):
        record = {
            "id": "s1",
            "question": "Select correct option",
            "choices": ["A1", "B2"],
            "answer": "B",
            "prompt": "legacy prompt",
            "task_id": "piqa_eval",
            "version_id": "v1",
        }
        sample = convert_llmeval_record(record, dataset_id="piqa")
        self.assertEqual(sample["_dataset_id"], "piqa")
        self.assertEqual(sample["metadata"]["correct_choice"], "B")
        self.assertEqual(sample["messages"][0]["content"][0]["text"], "Select correct option")
        self.assertEqual(sample["choices"][1]["message"]["content"][0]["text"], "B2")
        self.assertEqual(sample["inputs"]["prompt"], "legacy prompt")
        self.assertEqual(sample["audit_info"]["task_id"], "piqa_eval")
        self.assertEqual(sample["cache_suffix"], "-converted")
        self.assertEqual(sample["template_source"], "llm-eval")

    def test_multimodal_content_root_and_media_meta(self):
        record = {
            "id": "m1",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "q"},
                        {"type": "image_url", "image_url": {"url": "img/a.png", "detail": "hi-res"}},
                    ],
                }
            ],
        }
        sample = convert_llmeval_record(
            record,
            dataset_id="mmmu",
            content_field="messages.0.content",
            content_root="/root",
        )
        inputs = sample.get("inputs") or {}
        self.assertEqual(inputs["multi_modal_data"]["image"][0], "/root/img/a.png")
        self.assertIn("_media_meta", sample)
        self.assertEqual(sample["_media_meta"]["images"][0]["url"], "/root/img/a.png")
        self.assertEqual(sample["metadata"]["content_root"], "/root")
        self.assertEqual(sample["metadata"]["content_field"], "messages.0.content")


if __name__ == "__main__":
    unittest.main()
