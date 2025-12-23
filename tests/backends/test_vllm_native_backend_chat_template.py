import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_native_backend import VLLMNativeBackend


class DummyModel:
    def generate(self, prompts, sampling_params=None):
        class Out:
            outputs = [type("o", (), {"text": prompts[0]})]

        return [Out()]


def make_backend(config):
    orig_load = VLLMNativeBackend.load_model
    VLLMNativeBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = VLLMNativeBackend(config)
    finally:
        VLLMNativeBackend.load_model = orig_load
    return backend


class VLLMNativeBackendChatTemplateTests(unittest.TestCase):
    def test_template_fn_preferred_when_available(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"})

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "templated"

        backend._tokenizer = FakeTokenizer()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prepared = backend.prepare_inputs(payload)
        self.assertEqual(prepared["prompt"], "templated")
        self.assertEqual(prepared.get("cache_suffix"), "-chat_template")

    def test_never_mode_plain_render(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "never"})
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prepared = backend.prepare_inputs(payload)
        self.assertIn("assistant:", prepared["prompt"])
        self.assertIn("user: hi", prepared["prompt"])
        self.assertEqual(prepared.get("cache_suffix"), "-plain")

    def test_preprocessed_prompt_not_rerendered(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"})
        payload = {"prompt": "ready", "chat_template_mode": "preprocess", "messages": [{"role": "user", "content": "hi"}]}
        prepared = backend.prepare_inputs(payload)
        self.assertEqual(prepared["prompt"], "ready")

    def test_init_tokenizer_falls_back_to_model_path(self):
        backend = make_backend({"model_path": "repo"})
        import types

        called = {}

        class DummyTokenizer:
            @classmethod
            def from_pretrained(cls, name, **kwargs):
                called["name"] = name
                return cls()

        fake_module = types.SimpleNamespace(AutoTokenizer=DummyTokenizer)
        sys.modules["transformers"] = fake_module
        try:
            tok = backend._init_tokenizer({"model_path": "repo"})
        finally:
            sys.modules.pop("transformers", None)
        self.assertIsInstance(tok, DummyTokenizer)
        self.assertEqual(called.get("name"), "repo")


if __name__ == "__main__":
    unittest.main()
