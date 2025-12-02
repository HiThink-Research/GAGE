import sys
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.runtime.chat_template_mixin import (
    BackendCapabilities,
    ChatTemplateMixin,
    ChatTemplatePolicy,
)


def _fallback(messages):
    return "\n".join(f"{m.get('role')}: {m.get('content')}" for m in messages or [])


class ChatTemplateMixinTests(unittest.TestCase):
    def test_detect_multimodal_with_messages(self):
        payload = {"messages": [{"role": "user", "content": [{"type": "image", "image": "img.png"}]}]}
        self.assertTrue(ChatTemplateMixin.detect_multimodal(payload))

    def test_detect_multimodal_with_mm_data(self):
        payload = {"multi_modal_data": {"image": ["a.png"]}}
        self.assertTrue(ChatTemplateMixin.detect_multimodal(payload))

    def test_detect_multimodal_false(self):
        payload = {"messages": [{"role": "user", "content": "hi"}]}
        self.assertFalse(ChatTemplateMixin.detect_multimodal(payload))

    def test_should_render_respects_preprocess_and_plain(self):
        policy = ChatTemplatePolicy(mode="auto")
        payload = {"chat_template_mode": "preprocess"}
        self.assertFalse(ChatTemplateMixin.should_render(payload, policy))
        payload = {"chat_template_mode": "plain"}
        self.assertFalse(ChatTemplateMixin.should_render(payload, policy))
        payload = {"rendered_by": "preprocess"}
        self.assertFalse(ChatTemplateMixin.should_render(payload, policy))

    def test_select_template_prefers_model_for_vlm_when_available(self):
        policy = ChatTemplatePolicy(mode="auto", source="model")
        caps = BackendCapabilities(supports_mm=True, has_processor_chat_template=True)
        self.assertEqual(ChatTemplateMixin.select_template("vlm", policy, caps), "model")
        caps.has_processor_chat_template = False
        self.assertEqual(ChatTemplateMixin.select_template("vlm", policy, caps), "fallback")

    def test_render_uses_template_fn_then_fallback(self):
        calls = {}

        def template_fn(msgs, **kwargs):
            calls["called"] = True
            return "ok"

        out = ChatTemplateMixin.render([{"role": "user", "content": "hi"}], template_fn, _fallback)
        self.assertEqual(out, "ok")
        self.assertTrue(calls.get("called"))

        # template raises -> fallback
        def bad_template(msgs, **kwargs):
            raise RuntimeError("boom")

        out = ChatTemplateMixin.render([{"role": "user", "content": "hi"}], bad_template, _fallback)
        self.assertIn("user", out)

    def test_get_cache_suffix(self):
        policy = ChatTemplatePolicy(mode="never")
        caps = BackendCapabilities()
        self.assertEqual(ChatTemplateMixin.get_cache_suffix("text", policy, caps), "-plain")
        policy = ChatTemplatePolicy(mode="auto", source="model")
        self.assertEqual(ChatTemplateMixin.get_cache_suffix("text", policy, caps), "-chat_template")
        caps.has_processor_chat_template = True
        self.assertEqual(ChatTemplateMixin.get_cache_suffix("vlm", policy, caps), "-processor")
        policy = ChatTemplatePolicy(mode="auto", source="fallback")
        self.assertEqual(ChatTemplateMixin.get_cache_suffix("vlm", policy, caps), "-fallback")


if __name__ == "__main__":
    unittest.main()
