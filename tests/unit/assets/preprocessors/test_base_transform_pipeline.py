import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor


class _EchoPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        sample = dict(record)
        sample["messages"] = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        return sample


class _EmptyPre(BasePreprocessor):
    def to_sample(self, record, **kwargs):
        return dict(record)


class BaseTransformPipelineTests(unittest.TestCase):
    def test_doc_to_hooks_and_merge(self):
        pre = _EchoPre()
        sample = {"id": "s1"}
        out = pre.transform(sample, doc_to_visual=lambda rec: [{"type": "image_url", "image_url": {"url": "img.png"}}])
        self.assertIsInstance(out, dict)
        self.assertEqual(sample.get("visual")[0]["image_url"]["url"], "img.png")
        self.assertEqual(sample["_dataset_id"], "unknown")

    def test_on_error_skip(self):
        class _Boom(BasePreprocessor):
            def to_sample(self, record, **kwargs):
                raise ValueError("boom")

        sample = {"id": "s2"}
        pre = _Boom(on_error="skip")
        self.assertIsNone(pre.transform(sample))
        self.assertEqual(sample.get("metadata", {}).get("preprocess_error"), "boom")

    def test_doc_to_text_runs_when_empty(self):
        pre = _EmptyPre()
        sample = {"id": "s3"}
        out = pre.transform(sample, doc_to_text=lambda rec: "text-payload")
        self.assertIsInstance(out, dict)
        self.assertEqual(sample.get("text"), "text-payload")


if __name__ == "__main__":
    unittest.main()
