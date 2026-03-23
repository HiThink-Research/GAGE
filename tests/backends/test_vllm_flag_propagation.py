from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.vllm_backend import VLLMBackend


class DummyModel:
    """Lightweight placeholder to avoid real vLLM engine initialization in tests."""


def make_backend() -> VLLMBackend:
    """Create a backend instance with heavyweight dependencies patched out."""

    config = {"model_path": "dummy-model", "tokenizer_path": "dummy-tokenizer"}
    with (
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend.load_model", return_value=DummyModel()),
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend._init_tokenizer", return_value=MagicMock()),
        patch("gage_eval.role.model.backends.vllm_backend.VLLMBackend._load_auto_processor", return_value=MagicMock()),
    ):
        backend = VLLMBackend(config)
    backend._cfg_tokenizer_path = "dummy-tokenizer"
    return backend


class VLLMFlagPropagationTests(unittest.TestCase):
    def test_rendered_by_flag_propagates_from_sample_metadata(self):
        backend = make_backend()
        payload = {
            "sample": {
                "prompt": "PRE_RENDERED_PROMPT",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {
                    "rendered_by": "preprocess",
                    "chat_template_mode": "preprocess",
                },
            }
        }

        with patch("gage_eval.role.model.backends.vllm_backend.check_tokenizer_conflict"):
            prepared = backend.prepare_inputs(payload)

        self.assertEqual(prepared["rendered_by"], "preprocess")
        self.assertEqual(prepared["chat_template_mode"], "preprocess")
        self.assertEqual(prepared["prompt"], "PRE_RENDERED_PROMPT")

    def test_prepare_inputs_renders_prompt_when_preprocess_flags_are_absent(self):
        backend = make_backend()
        payload = {
            "sample": {
                "prompt": "RAW_INPUT",
                "messages": [{"role": "user", "content": "hello"}],
                "metadata": {},
            }
        }

        with (
            patch("gage_eval.role.model.backends.vllm_backend.check_tokenizer_conflict"),
            patch.object(backend, "_render_prompt", return_value="RENDERED_BY_BACKEND") as render_prompt,
            patch.object(backend, "_maybe_tokenize_messages", return_value=("RENDERED_BY_BACKEND", {})),
        ):
            prepared = backend.prepare_inputs(payload)

        render_prompt.assert_called_once()
        self.assertNotEqual(prepared.get("rendered_by"), "preprocess")
        self.assertEqual(prepared["prompt"], "RENDERED_BY_BACKEND")


if __name__ == "__main__":
    unittest.main()
