import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_backend import VLLMBackend


class DummyModel:
    pass


def make_backend(config):
    """Create backend without loading real vLLM."""

    orig_load = VLLMBackend.load_model
    VLLMBackend.load_model = lambda self, cfg: DummyModel()
    try:
        backend = VLLMBackend(config)
    finally:
        VLLMBackend.load_model = orig_load
    return backend


class VLLMBackendChatTemplateTests(unittest.TestCase):
    def test_template_fn_preferred_when_available(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"})

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "templated"

        backend._tokenizer = FakeTokenizer()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertEqual(out["prompt"], "templated")
        self.assertEqual(out.get("cache_suffix"), "-chat_template")

    def test_never_mode_falls_back_to_plain(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "never"})
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertIn("assistant:", out["prompt"])
        self.assertIn("user: hi", out["prompt"])
        self.assertEqual(out.get("cache_suffix"), "-plain")

    def test_preprocessed_prompt_not_rerendered(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"})
        payload = {"prompt": "ready", "chat_template_mode": "preprocess", "messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertEqual(out["prompt"], "ready")

    def test_preprocess_fallback_allows_backend_rerender(self):
        backend = make_backend({"model_path": "repo", "use_chat_template": "auto"})

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                return "templated_backend"

        backend._tokenizer = FakeTokenizer()
        payload = {
            "messages": [{"role": "user", "content": "hi"}],
            "prompt": "fallback",
            "chat_template_mode": "preprocess",
            "template_source": "fallback",
            "rendered_by": "preprocess",
        }
        out = backend.prepare_inputs(payload)
        self.assertEqual(out["prompt"], "templated_backend")
        self.assertEqual(out.get("cache_suffix"), "-chat_template")

    def test_init_tokenizer_falls_back_to_model_path(self):
        backend = make_backend({"model_path": "repo"})
        # Patch transformers.AutoTokenizer
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
