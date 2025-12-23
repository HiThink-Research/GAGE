import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vlm_transformers_backend import VLMTransformersBackend


class DummyModel:
    pass


def make_backend():
    # minimal config
    cfg = {
        "model_name_or_path": "repo",
        "processor_name_or_path": "repo",
        "trust_remote_code": True,
        "use_chat_template_vlm": "auto",
    }
    orig_load = VLMTransformersBackend.load_model
    VLMTransformersBackend.load_model = lambda self, c: DummyModel()
    try:
        backend = VLMTransformersBackend(cfg)
    finally:
        VLMTransformersBackend.load_model = orig_load
    # 默认 processor 占位，避免 AttributeError
    backend.processor = type("P", (), {})()
    return backend


class VLMTransformersBackendChatTemplateTests(unittest.TestCase):
    def test_processor_template_applied(self):
        backend = make_backend()

        class FakeProcessor:
            def apply_chat_template(self, messages, **kwargs):
                return "templated"

        backend.processor = FakeProcessor()
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prompt = backend._render_messages(payload["messages"], payload)
        self.assertEqual(prompt, "templated")
        self.assertEqual(payload.get("cache_suffix"), "-processor")

    def test_force_never_plain(self):
        backend = make_backend()
        backend._chat_template_policy.mode = "never"
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        prompt = backend._render_messages(payload["messages"], payload)
        self.assertIn("assistant:", prompt)
        self.assertEqual(payload.get("cache_suffix"), "-plain")

    def test_preprocessed_not_rerender(self):
        backend = make_backend()
        payload = {"chat_template_mode": "preprocess", "messages": [{"role": "user", "content": "hi"}]}
        # should skip rendering and fall back to simple render only if needed; here we expect plain render due to skip
        prompt = backend._render_messages(payload["messages"], payload)
        self.assertIn("assistant:", prompt)


if __name__ == "__main__":
    unittest.main()
