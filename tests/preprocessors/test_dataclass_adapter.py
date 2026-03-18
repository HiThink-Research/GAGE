import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.sample import sample_from_dict, sample_to_dict, append_prediction


class DataclassAdapterTests(unittest.TestCase):
    def test_roundtrip_and_append_prediction(self):
        raw = {
            "id": "s1",
            "_dataset_id": "piqa",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [{"index": 0, "message": {"role": "assistant", "content": [{"type": "text", "text": "hello"}]}}],
            "metadata": {"option_map": {"A": "hello"}, "correct_choice": "A"},
            "inputs": {"prompt": "hi"},
        }
        sample = sample_from_dict(raw)
        self.assertEqual(sample.messages[0].content[0].text, "hi")

        append_prediction(sample, {"answer": "A"})
        self.assertEqual(len(sample.predict_result), 1)
        self.assertEqual(sample.predict_result[0].message.content[0].text, "A")

        back = sample_to_dict(sample)
        self.assertEqual(back["id"], "s1")
        self.assertEqual(back["predict_result"][0]["message"]["content"][0]["text"], "A")

    def test_string_message_content_stays_as_one_fragment(self):
        raw = {
            "id": "s2",
            "messages": [{"role": "user", "content": "hello world"}],
        }

        sample = sample_from_dict(raw)

        self.assertEqual(len(sample.messages[0].content), 1)
        self.assertEqual(sample.messages[0].content[0].text, "hello world")

    def test_dict_message_content_is_wrapped(self):
        raw = {
            "id": "s3",
            "messages": [{"role": "user", "content": {"type": "text", "text": "hello"}}],
        }

        sample = sample_from_dict(raw)

        self.assertEqual(len(sample.messages[0].content), 1)
        self.assertEqual(sample.messages[0].content[0].text, "hello")


if __name__ == "__main__":
    unittest.main()
