import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


class _FakeLitellm(types.SimpleNamespace):
    def __init__(self, *, raise_error: bool = False):
        super().__init__()
        self.calls = []
        self.raise_error = raise_error
        self.drop_params = False
        self.verbose = False
        self.api_key = None
        self.api_base = None
        self.headers = None

    def completion(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_error:
            raise RuntimeError("litellm failure")
        return {"choices": [{"message": {"content": "pong-lite"}}]}

    def supports_reasoning(self, _model):
        return False


class LiteLLMBackendTests(unittest.TestCase):
    def test_litellm_backend_merges_sampling_and_headers(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16, "stop": ["END"]},
                    "extra_headers": {"X-Test": "1"},
                    "timeout": 5.0,
                }
            )
            result = backend.generate(
                {
                    "messages": [{"role": "user", "content": "ping"}],
                    "sampling_params": {"temperature": 0.1, "max_new_tokens": 8},
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["temperature"], 0.1)
        self.assertEqual(call["max_tokens"], 8)
        self.assertEqual(call["stop"], ["END"])
        self.assertEqual(call["headers"]["X-Test"], "1")

    def test_grok_defaults_to_xai_base(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "grok-1.5",
                    "api_key": "xai-key",
                    "generation_parameters": {"max_new_tokens": 32},
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "hello grok"}]})

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["base_url"], "https://api.x.ai/v1")
        self.assertEqual(call["api_base"], "https://api.x.ai/v1")
        self.assertEqual(call["api_key"], "xai-key")
        self.assertEqual(call["custom_llm_provider"], "xai")

    def test_kimi_uses_litellm_without_http_fallback(self):
        fake_litellm = _FakeLitellm(raise_error=True)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "moonshot-v1-8k",
                    "provider": "kimi",
                    "api_key": "kimi-key",
                    "retry_sleep": 0.01,
                    "max_retries": 2,
                }
            )
            with self.assertRaises(RuntimeError):
                backend.generate({"messages": [{"role": "user", "content": "hello"}], "sampling_params": {"max_new_tokens": 5}})

        self.assertEqual(len(fake_litellm.calls), 2, "LiteLLM retries should be honored without HTTP fallback")
        call = fake_litellm.calls[0]
        self.assertIn("moonshot", call["base_url"])
        self.assertEqual(call["api_key"], "kimi-key")
        self.assertEqual(call["custom_llm_provider"], "moonshot")

    def test_azure_easy_config_fills_base_and_version(self):
        fake_litellm = _FakeLitellm()
        env = {
            "AZURE_OPENAI_ENDPOINT": "https://demo-openai.eastus.azure.com",
            "AZURE_OPENAI_API_KEY": "azure-key",
            "AZURE_OPENAI_API_VERSION": "2024-06-01-preview",
            "OPENAI_API_KEY": "",  # Unset to prevent interference from local environment
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "azure:gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "hello azure"}]})

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["base_url"], "https://demo-openai.eastus.azure.com")
        self.assertEqual(call["api_key"], "azure-key")
        self.assertEqual(call["api_version"], "2024-06-01-preview")
        self.assertEqual(call["api_type"], "azure")
        self.assertEqual(call["custom_llm_provider"], "azure")


if __name__ == "__main__":
    unittest.main()
