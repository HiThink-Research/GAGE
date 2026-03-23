import asyncio
import os
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


class _FakeLitellm(types.SimpleNamespace):
    def __init__(self, *, raise_error: bool = False, response: object | None = None):
        super().__init__()
        self.calls = []
        self.raise_error = raise_error
        self.response = response or {"choices": [{"message": {"content": "pong-lite"}}]}
        self.drop_params = False
        self.verbose = False
        self.api_key = None
        self.api_base = None
        self.headers = None

    def completion(self, **kwargs):
        payload = dict(kwargs)
        payload["_drop_params"] = self.drop_params
        payload["_verbose"] = self.verbose
        self.calls.append(payload)
        if self.raise_error:
            raise RuntimeError("litellm failure")
        return self.response

    def supports_reasoning(self, _model):
        return False


class LiteLLMBackendTests(unittest.TestCase):
    def test_credentials_are_scoped_to_each_request_not_litellm_globals(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend_a = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "api_key": "key-a",
                    "api_base": "https://api-a.example/v1",
                    "extra_headers": {"Authorization": "Bearer alpha"},
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            backend_b = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "api_key": "key-b",
                    "api_base": "https://api-b.example/v1",
                    "extra_headers": {"Authorization": "Bearer beta"},
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )

            backend_a.generate({"messages": [{"role": "user", "content": "ping-a"}]})
            backend_b.generate({"messages": [{"role": "user", "content": "ping-b"}]})

        self.assertIsNone(fake_litellm.api_key)
        self.assertIsNone(fake_litellm.api_base)
        self.assertIsNone(fake_litellm.headers)
        self.assertEqual(fake_litellm.calls[0]["api_key"], "key-a")
        self.assertEqual(fake_litellm.calls[0]["api_base"], "https://api-a.example/v1")
        self.assertEqual(fake_litellm.calls[0]["headers"]["Authorization"], "Bearer alpha")
        self.assertEqual(fake_litellm.calls[1]["api_key"], "key-b")
        self.assertEqual(fake_litellm.calls[1]["api_base"], "https://api-b.example/v1")
        self.assertEqual(fake_litellm.calls[1]["headers"]["Authorization"], "Bearer beta")

    def test_litellm_backend_does_not_mutate_module_flags_during_init(self):
        fake_litellm = _FakeLitellm()
        fake_litellm.drop_params = "keep-drop"
        fake_litellm.verbose = "keep-verbose"
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            LiteLLMBackend({"model": "gpt-4o-mini"})

        self.assertEqual(fake_litellm.drop_params, "keep-drop")
        self.assertEqual(fake_litellm.verbose, "keep-verbose")

    def test_litellm_backend_restores_module_flags_after_request(self):
        fake_litellm = _FakeLitellm()
        fake_litellm.drop_params = "persist-drop"
        fake_litellm.verbose = "persist-verbose"
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend({"model": "gpt-4o-mini", "verbose": True})
            backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(fake_litellm.calls[0]["_drop_params"], True)
        self.assertEqual(fake_litellm.calls[0]["_verbose"], True)
        self.assertEqual(fake_litellm.drop_params, "persist-drop")
        self.assertEqual(fake_litellm.verbose, "persist-verbose")

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

    def test_wrapped_litellm_backend_preserves_native_retry_budget(self):
        fake_litellm = _FakeLitellm(raise_error=True)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "moonshot-v1-8k",
                    "provider": "kimi",
                    "api_key": "kimi-key",
                    "retry_sleep": 0.0,
                    "max_retries": 2,
                }
            )
            wrapped = wrap_backend(backend)

            result = asyncio.run(
                wrapped.ainvoke(
                    {"messages": [{"role": "user", "content": "hello"}], "sampling_params": {"max_new_tokens": 5}}
                )
            )

        self.assertEqual(result["error"], "litellm failure")
        self.assertEqual(result["backend"], "LiteLLMBackend")
        self.assertEqual(len(fake_litellm.calls), 2)

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

    def test_deepseek_flattens_image_messages_and_uses_native_provider(self):
        fake_litellm = _FakeLitellm()
        env = {
            "DEEPSEEK_API_KEY": "deepseek-key",
            "OPENAI_API_KEY": "",
        }
        with mock.patch.dict(os.environ, env, clear=False), mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "provider": "openai",
                    "model": "deepseek-chat",
                    "api_base": "https://api.deepseek.com",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            prepared = backend.prepare_inputs(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "Describe this image briefly."},
                                {"type": "image_url", "image_url": {"url": "https://example.com/demo.png"}},
                            ],
                        }
                    ]
                }
            )
            result = backend.generate(prepared)

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["model"], "deepseek/deepseek-chat")
        self.assertEqual(call["api_key"], "deepseek-key")
        self.assertEqual(call["custom_llm_provider"], "deepseek")
        self.assertIsInstance(call["messages"][0]["content"], str)
        self.assertIn("<image>", call["messages"][0]["content"])

    def test_openai_multimodal_messages_keep_image_blocks(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "api_key": "openai-key",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            prepared = backend.prepare_inputs(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "What is shown?"},
                                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                            ],
                        }
                    ]
                }
            )
            result = backend.generate(prepared)

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertIsInstance(call["messages"][0]["content"], list)
        self.assertEqual(call["messages"][0]["content"][1]["type"], "image_url")
        self.assertEqual(call["messages"][0]["content"][1]["image_url"]["url"], "https://example.com/image.png")

    def test_non_deepseek_target_does_not_use_deepseek_api_key(self):
        """Non-DeepSeek targets should ignore the DeepSeek-specific API key."""

        fake_litellm = _FakeLitellm()
        with mock.patch.dict(
            os.environ,
            {
                "DEEPSEEK_API_KEY": "deepseek-key",
                "OPENAI_API_KEY": "openai-key",
            },
            clear=True,
        ), mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["api_key"], "openai-key")

    def test_thinking_mode_disabled_injects_enable_thinking_false(self):
        """Thinking mode 'disabled' should inject enable_thinking=False into litellm kwargs."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "qwen3-32b",
                    "thinking_mode": "disabled",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "solve 2+2"}], "sampling_params": {"max_new_tokens": 16}}
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertIn("enable_thinking", call)
        self.assertFalse(call["enable_thinking"])

    def test_thinking_mode_enabled_injects_enable_thinking_true(self):
        """Thinking mode 'enabled' should inject enable_thinking=True into litellm kwargs."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "qwen3-32b",
                    "thinking_mode": "enabled",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "solve 2+2"}], "sampling_params": {"max_new_tokens": 16}}
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertIn("enable_thinking", call)
        self.assertTrue(call["enable_thinking"])

    def test_reasoning_effort_forwarded_in_kwargs(self):
        """reasoning_effort from generation_parameters should be forwarded to litellm."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o",
                    "generation_parameters": {"max_new_tokens": 16, "reasoning_effort": "high"},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "think hard"}], "sampling_params": {"max_new_tokens": 16}}
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call.get("reasoning_effort"), "high")

    def test_no_thinking_mode_does_not_inject_enable_thinking(self):
        """When thinking_mode is None, enable_thinking should not appear in kwargs."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "ping"}], "sampling_params": {"max_new_tokens": 16}}
            )

        call = fake_litellm.calls[0]
        self.assertNotIn("enable_thinking", call)
        self.assertNotIn("reasoning_effort", call)

    def test_response_logging_uses_safe_summary_without_answer_content(self):
        fake_litellm = _FakeLitellm(
            response={
                "id": "resp-1",
                "object": "chat.completion",
                "model": "gpt-4o-mini",
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "content": "super-secret-answer",
                            "tool_calls": [{"id": "tool-1"}],
                        },
                    }
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8},
            }
        )
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            with mock.patch("gage_eval.role.model.backends.litellm_backend.logger.info") as mock_logger_info:
                result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(result["answer"], "super-secret-answer")
        message, payload = mock_logger_info.call_args_list[-1].args
        self.assertEqual(message, "LiteLLM response summary: {}")
        self.assertNotIn("super-secret-answer", payload)
        self.assertNotIn("tool-1", payload)
        self.assertIn("\"answer_chars\": 19", payload)
        self.assertIn("\"has_tool_calls\": true", payload)
        self.assertIn("\"finish_reason\": \"stop\"", payload)


if __name__ == "__main__":
    unittest.main()
