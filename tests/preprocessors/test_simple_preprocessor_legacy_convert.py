import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.simple_preprocessor import SimplePreprocessor


class _LegacyMock(SimplePreprocessor):
    def to_legacy(self, record, **kwargs):
        return {
            "id": "l1",
            "question": "Q?",
            "choices": ["opt1", "opt2"],
            "answer": "B",
            "prompt": "prompt-text",
        }


class SimplePreprocessorLegacyConvertTests(unittest.TestCase):
    def test_convert_llmeval_record(self):
        pre = _LegacyMock()
        sample = {}
        out = pre.transform(sample, dataset_id="ds1")
        self.assertEqual(out, {"prompt": "prompt-text"})
        self.assertEqual(sample["_dataset_id"], "ds1")
        self.assertEqual(sample["metadata"]["correct_choice"], "B")
        self.assertEqual(sample["choices"][1]["message"]["content"][0]["text"], "opt2")
        self.assertEqual(sample["chat_template_mode"], "converted")
        self.assertEqual(sample["cache_suffix"], "-converted")


if __name__ == "__main__":
    unittest.main()
