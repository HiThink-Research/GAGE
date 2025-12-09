import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.loaders.loader_utils import apply_preprocess
from gage_eval.config.pipeline_config import DatasetSpec


class DummyTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return "templated"


class DefaultChatTemplatePreprocessTests(unittest.TestCase):
    def test_auto_preprocess_applies_template(self):
        spec = DatasetSpec(
            dataset_id="d1",
            loader="jsonl",
            params={"tokenizer_path": "repo"},
        )
        records = [
            {"id": 1, "messages": [{"role": "user", "content": "hi"}]},
        ]

        from gage_eval.assets.datasets import loaders
        loader_module = loaders.loader_utils
        orig_load = loader_module.load_tokenizer
        orig_get = loader_module.get_or_load_tokenizer
        loader_module._TOKENIZER_MANAGER._cache.clear()
        dummy = DummyTokenizer()
        loader_module.load_tokenizer = lambda args: dummy
        loader_module.get_or_load_tokenizer = lambda args: dummy
        try:
            processed = list(apply_preprocess(records, spec, data_path=None))
        finally:
            loader_module.load_tokenizer = orig_load
            loader_module.get_or_load_tokenizer = orig_get

        self.assertEqual(len(processed), 1)
        item = processed[0]
        self.assertEqual(item["inputs"], {"prompt": "templated"})
        self.assertEqual(item["prompt"], "templated")
        self.assertEqual(item.get("chat_template_mode"), "preprocess")
        self.assertEqual(item.get("template_source"), "model")
        self.assertEqual(item.get("rendered_by"), "preprocess")
        self.assertEqual(item.get("cache_suffix"), "-chat_template")
        self.assertEqual(dummy.calls, 1)

    def test_env_fallback_tokenizer_name(self):
        spec = DatasetSpec(
            dataset_id="d1",
            loader="jsonl",
            params={},
        )
        records = [{"id": 1, "messages": [{"role": "user", "content": "hi"}]}]

        from gage_eval.assets.datasets import loaders
        loader_module = loaders.loader_utils
        orig_load = loader_module.load_tokenizer
        orig_get = loader_module.get_or_load_tokenizer
        loader_module._TOKENIZER_MANAGER._cache.clear()
        loader_module.load_tokenizer = lambda args: DummyTokenizer()
        loader_module.get_or_load_tokenizer = lambda args: DummyTokenizer()
        import os

        os.environ["GAGE_EVAL_MODEL_PATH"] = "repo_env"
        try:
            processed = list(apply_preprocess(records, spec, data_path=None))
        finally:
            loader_module.load_tokenizer = orig_load
            loader_module.get_or_load_tokenizer = orig_get
            os.environ.pop("GAGE_EVAL_MODEL_PATH", None)
            os.environ.pop("MODEL_PATH", None)

        self.assertEqual(processed[0]["inputs"], {"prompt": "templated"})
        self.assertEqual(processed[0]["_tokenizer_path"], "repo_env")

    def test_list_content_normalized_then_template(self):
        spec = DatasetSpec(
            dataset_id="d1",
            loader="jsonl",
            params={"tokenizer_path": "repo"},
        )
        records = [
            {"id": 1, "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}, {"type": "text", "text": "there"}]}]},
        ]

        from gage_eval.assets.datasets import loaders
        loader_module = loaders.loader_utils
        orig_load = loader_module.load_tokenizer
        orig_get = loader_module.get_or_load_tokenizer
        loader_module._TOKENIZER_MANAGER._cache.clear()
        dummy = DummyTokenizer()
        loader_module.load_tokenizer = lambda args: dummy
        loader_module.get_or_load_tokenizer = lambda args: dummy
        try:
            processed = list(apply_preprocess(records, spec, data_path=None))
        finally:
            loader_module.load_tokenizer = orig_load
            loader_module.get_or_load_tokenizer = orig_get

        item = processed[0]
        self.assertEqual(item["inputs"], {"prompt": "templated"})
        self.assertEqual(dummy.calls, 1)

    def test_skip_when_no_messages(self):
        spec = DatasetSpec(dataset_id="d1", loader="jsonl", params={"tokenizer_path": "repo"})
        records = [{"id": 2, "prompt": "hi"}]
        processed = list(apply_preprocess(records, spec, data_path=None))
        self.assertIn("inputs", processed[0])
        self.assertEqual(processed[0]["inputs"].get("prompt"), "hi")


if __name__ == "__main__":
    unittest.main()
