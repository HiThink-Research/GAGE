"""Regression test for multimodal vision token duplication bug.

This test verifies that for Qwen2-VL style models, vision tokens are only
inserted once, not twice (which would cause IndexError in vLLM's rotary_embedding).
"""

import unittest
from unittest.mock import MagicMock, patch
import sys
sys.path.insert(0, "src")


class TestMultimodalVisionTokenDuplication(unittest.TestCase):
    """Test that multimodal requests don't get double vision token rendering."""

    def test_prepare_inputs_skips_rendering_for_multimodal(self):
        """Verify prepare_inputs doesn't render for multimodal requests."""
        from gage_eval.role.model.backends.vllm_backend import VLLMBackend
        from gage_eval.role.common.backend_utils import has_multimodal_inputs
        
        # Create a mock backend
        with patch.object(VLLMBackend, '__init__', lambda self, **kwargs: None):
            backend = VLLMBackend()
            backend._tokenizer = MagicMock()
            backend._processor = MagicMock()
            backend._mm_supported = True
            backend._mm_strategy = "inputs"
            backend._chat_template_mode = "auto"
            backend._chat_template_policy = MagicMock()
            backend._cfg_tokenizer_path = None
            backend._fallback_template = None
            backend._force_tokenize_prompt = False
            backend._default_sampling = {}
            backend._max_tokens = 512

            # Create multimodal payload
            payload = {
                "sample": {
                    "messages": [
                        {"role": "user", "content": [
                            {"type": "text", "text": "What is this?"},
                            {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
                        ]}
                    ],
                    "metadata": {}
                }
            }

            prepared = backend.prepare_inputs(payload)
            
            # For multimodal, prompt should be empty or just raw text (no chat template applied)
            # The actual rendering happens in _async_generate
            prompt = prepared.get("prompt", "")
            
            # Should NOT contain Qwen vision tokens yet - those are added in _async_generate
            self.assertNotIn("<|vision_start|>", prompt, 
                "prepare_inputs should NOT render with processor for multimodal")
            self.assertNotIn("<|image_pad|>", prompt,
                "prepare_inputs should NOT render with processor for multimodal")

    def test_multimodal_detection(self):
        """Verify has_multimodal_inputs correctly detects image_url in messages."""
        from gage_eval.role.common.backend_utils import has_multimodal_inputs
        
        # With image_url
        prepared_mm = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "test"},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,test"}}
                ]}
            ]
        }
        self.assertTrue(has_multimodal_inputs(prepared_mm), 
            "Should detect image_url in messages")
        
        # Without image
        prepared_text = {
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "test"}
                ]}
            ]
        }
        self.assertFalse(has_multimodal_inputs(prepared_text),
            "Should not detect multimodal for text-only")

    def test_render_flags_propagation(self):
        """Verify render flags are correctly extracted from sample.metadata."""
        from gage_eval.role.model.backends.vllm.vllm_request import _collect_chat_meta
        
        sample = {
            "metadata": {
                "chat_template_mode": "preprocess",
                "rendered_by": "preprocess",
            }
        }
        payload = {"sample": sample}
        
        chat_meta = _collect_chat_meta(payload, sample)
        
        self.assertEqual(chat_meta.get("chat_template_mode"), "preprocess",
            "Should extract chat_template_mode from metadata")
        self.assertEqual(chat_meta.get("rendered_by"), "preprocess",
            "Should extract rendered_by from metadata")


if __name__ == "__main__":
    unittest.main()
