import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.legacy_vllm_backend import LegacyVLLMBackend


class DummyEngine:
    def __init__(self):
        self.calls = []

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        return {"outputs": [{"text": "ok"}]}


class FakeProcessor:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return "processor_templated"


class EmptyListProcessor:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return []


class TrackingProcessor:
    def __init__(self):
        self.calls = 0

    def apply_chat_template(self, messages, **kwargs):
        self.calls += 1
        return "unused"


def make_backend(processor=None):
    orig_load = LegacyVLLMBackend.load_model
    orig_init_tok = LegacyVLLMBackend._init_tokenizer
    LegacyVLLMBackend.load_model = lambda self, cfg: DummyEngine()
    LegacyVLLMBackend._init_tokenizer = lambda self, cfg: None
    try:
        backend = LegacyVLLMBackend({"model_path": "repo"})
        backend._processor = processor
    finally:
        LegacyVLLMBackend.load_model = orig_load
        LegacyVLLMBackend._init_tokenizer = orig_init_tok
    return backend


class MultimodalProcessorChatTemplateTests(unittest.TestCase):
    def test_processor_template_used_for_multimodal(self):
        processor = FakeProcessor()
        backend = make_backend(processor)
        payload = {
            "_tokenizer_path": "repo",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "hi"},
                        {"type": "image_url", "image_url": {"url": "img.png"}},
                    ],
                }
            ]
        }
        prepared = backend.prepare_inputs(payload)
        out = backend.generate(prepared)
        self.assertGreaterEqual(processor.calls, 1)
        calls = backend.model.calls
        self.assertEqual(len(calls), 1)
        call_kwargs = calls[0]
        prompt_used = call_kwargs.get("inputs", {}).get("prompt") or call_kwargs.get("prompt")
        self.assertEqual(prompt_used, "processor_templated")
        self.assertEqual(out["answer"], "ok")

    def test_render_with_processor_empty_messages_skips(self):
        processor = TrackingProcessor()
        backend = make_backend(processor)
        rendered = backend._render_with_processor([], "fallback", {})
        self.assertIsNone(rendered)
        self.assertEqual(processor.calls, 0)

    def test_render_with_processor_empty_list_falls_back(self):
        processor = EmptyListProcessor()
        backend = make_backend(processor)
        rendered = backend._render_with_processor([{"role": "user", "content": "hi"}], "fallback", {})
        self.assertEqual(rendered, "fallback")
        self.assertEqual(processor.calls, 1)


if __name__ == "__main__":
    unittest.main()
