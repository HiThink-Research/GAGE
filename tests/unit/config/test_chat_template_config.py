import unittest

import sys
from pathlib import Path

from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.config.vllm import VLLMBackendConfig
from gage_eval.role.model.config.vlm_transformers import VLMTransformersBackendConfig


class ChatTemplateConfigTests(unittest.TestCase):
    def test_vllm_chat_template_default(self):
        cfg = VLLMBackendConfig(model_path="repo")
        self.assertEqual(cfg.use_chat_template, "auto")

    def test_vllm_chat_template_invalid_value(self):
        with self.assertRaises(ValidationError):
            VLLMBackendConfig(model_path="repo", use_chat_template="maybe")  # type: ignore[arg-type]

    def test_vllm_mm_limit_config_accepts_positive_limits(self):
        cfg = VLLMBackendConfig(model_path="repo", limit_mm_per_prompt={"image": 2, "audio": 1})
        self.assertEqual(cfg.limit_mm_per_prompt, {"image": 2, "audio": 1})

    def test_vllm_mm_limit_config_rejects_non_positive_limits(self):
        with self.assertRaises(ValidationError):
            VLLMBackendConfig(model_path="repo", limit_mm_per_prompt={"image": 0})

    def test_vlm_chat_template_default(self):
        cfg = VLMTransformersBackendConfig(model_name_or_path="hf/model")
        self.assertEqual(cfg.use_chat_template_vlm, "auto")

    def test_vlm_chat_template_invalid_value(self):
        with self.assertRaises(ValidationError):
            VLMTransformersBackendConfig(
                model_name_or_path="hf/model", use_chat_template_vlm="maybe"  # type: ignore[arg-type]
            )


if __name__ == "__main__":
    unittest.main()
