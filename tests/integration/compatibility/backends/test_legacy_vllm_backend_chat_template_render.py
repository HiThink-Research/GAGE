import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyModel:
    pass


class FakeTokenizer:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return "templated"


def make_backend(config=None, tokenizer=None):
    orig_load = LegacyVLLMBackend.load_model
    orig_init_tok = LegacyVLLMBackend._init_tokenizer

    def fake_load(self, cfg):
        self._tokenizer = tokenizer
        return DummyModel()

    LegacyVLLMBackend.load_model = fake_load
    LegacyVLLMBackend._init_tokenizer = lambda self, cfg: tokenizer
    try:
        backend = LegacyVLLMBackend(config or {"model_path": "repo"})
    finally:
        LegacyVLLMBackend.load_model = orig_load
        LegacyVLLMBackend._init_tokenizer = orig_init_tok
    return backend


class LegacyVLLMBackendChatTemplateRenderTests(unittest.TestCase):
    def test_renders_with_tokenizer(self):
        tok = FakeTokenizer()
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"}, tokenizer=tok)
        self.assertIs(backend._tokenizer, tok)
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prepared = backend.prepare_inputs(payload)
        self.assertEqual(prepared["prompt"], "templated")
        self.assertEqual(prepared.get("template_source"), "model")
        self.assertEqual(prepared.get("rendered_by"), "backend")
        self.assertEqual(prepared.get("cache_suffix"), "-chat_template")
        self.assertEqual(tok.calls, 1)

    def test_skip_if_preprocessed(self):
        tok = FakeTokenizer()
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"}, tokenizer=tok)
        payload = {
            "prompt": "ready",
            "chat_template_mode": "preprocess",
            "messages": [{"role": "user", "content": "hi"}],
        }
        prepared = backend.prepare_inputs(payload)
        self.assertEqual(prepared["prompt"], "ready")
        self.assertEqual(tok.calls, 0)


if __name__ == "__main__":
    unittest.main()
