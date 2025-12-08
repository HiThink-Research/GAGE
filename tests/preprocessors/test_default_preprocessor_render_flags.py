import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor


class DummyTokenizer:
    def __init__(self, rendered: str = "templated"):
        self.rendered = rendered
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return self.rendered


class DefaultPreprocessorRenderFlagTests(unittest.TestCase):
    def test_render_and_flags(self):
        pre = DefaultPreprocessor(tokenizer=DummyTokenizer(), tokenizer_path="tok-path")
        sample = {"messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]}
        out = pre.transform(sample)
        self.assertEqual(out, {"prompt": "templated"})
        self.assertEqual(sample["prompt"], "templated")
        self.assertEqual(sample["chat_template_mode"], "preprocess")
        self.assertEqual(sample["template_source"], "model")
        self.assertEqual(sample["rendered_by"], "preprocess")
        self.assertEqual(sample["cache_suffix"], "-chat_template")
        self.assertEqual(sample["_tokenizer_path"], "tok-path")


if __name__ == "__main__":
    unittest.main()
