import sys
from pathlib import Path
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_backend import VLLMBackend


class DummyModel:
    def generate(self, *args, **kwargs):
        return "ok"


def make_backend(config, model=None):
    """Create backend without loading real vLLM."""

    orig_load = VLLMBackend.load_model
    VLLMBackend.load_model = lambda self, cfg: model or DummyModel()
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

    def test_force_tokenize_normalizes_dict_prompt_token_ids(self):
        backend = make_backend({"model_path": "repo", "force_tokenize_prompt": True})

        class FakeTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                if kwargs.get("tokenize"):
                    return {"input_ids": [101, 102, 103]}
                return "templated_with_ids"

        backend._tokenizer = FakeTokenizer()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        out = backend.prepare_inputs(payload)
        self.assertEqual(out["prompt"], "templated_with_ids")
        self.assertEqual(out["inputs"]["prompt_token_ids"], [101, 102, 103])

    def test_generate_normalizes_dict_prompt_token_ids_from_inputs(self):
        class RecordingModel(DummyModel):
            def __init__(self):
                self.last_args = None
                self.last_kwargs = None

            def generate(self, *args, **kwargs):
                self.last_args = args
                self.last_kwargs = kwargs
                return "ok"

        model = RecordingModel()
        backend = make_backend({"model_path": "repo"}, model=model)
        prepared = {
            "prompt": "hello",
            "messages": [{"role": "user", "content": "hello"}],
            "inputs": {
                "prompt": "hello",
                "prompt_token_ids": {"input_ids": [11, 12, 13]},
            },
        }

        backend._generate_one(prepared, sampling_params={"temperature": 0.0}, request_id="req_1")

        self.assertIsNotNone(model.last_kwargs)
        self.assertTrue(model.last_args)
        prompt_input = model.last_args[0]
        self.assertEqual(prompt_input["prompt_token_ids"], [11, 12, 13])
        self.assertNotIn("input_ids", prompt_input)
        self.assertEqual(model.last_kwargs["sampling_params"], {"temperature": 0.0})
        self.assertEqual(model.last_kwargs["request_id"], "req_1")

    def test_refresh_engine_mm_support_accepts_engine_before_model_assignment(self):
        backend = object.__new__(VLLMBackend)
        backend._engine_mm_support = None
        backend._model_supports_mm = True

        engine = object()
        with patch("gage_eval.role.model.backends.vllm_backend._prime_vllm_engine_renderer_state") as prime, patch(
            "gage_eval.role.model.backends.vllm_backend._detect_vllm_engine_multimodal_support",
            return_value=True,
        ) as detect:
            result = VLLMBackend._refresh_engine_mm_support(backend, require_mm_processor=False, engine=engine)

        prime.assert_called_once_with(engine, require_mm_processor=False)
        detect.assert_called_once_with(engine)
        self.assertTrue(result)
        self.assertTrue(backend._engine_mm_support)
        self.assertTrue(backend._model_supports_mm)


if __name__ == "__main__":
    unittest.main()
