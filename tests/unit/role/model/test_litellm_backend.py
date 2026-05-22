import asyncio
from concurrent.futures import ThreadPoolExecutor
import os
import sys
import threading
import types
import unittest
from unittest import mock

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.dut_model import DUTModelAdapter
from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend

pytestmark = pytest.mark.fast


class _FakeLitellm(types.SimpleNamespace):
    def __init__(
        self,
        *,
        raise_error: bool = False,
        response: object | None = None,
        supports_function_calling: bool | None = None,
        supports_function_calling_error: bool = False,
    ):
        super().__init__()
        self.calls = []
        self.async_calls = []
        self.raise_error = raise_error
        self.response = response or {"choices": [{"message": {"content": "pong-lite"}}]}
        self.supports_function_calling_value = supports_function_calling
        self.supports_function_calling_error = supports_function_calling_error
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

    async def acompletion(self, **kwargs):
        payload = dict(kwargs)
        payload["_drop_params"] = self.drop_params
        payload["_verbose"] = self.verbose
        self.async_calls.append(payload)
        if self.raise_error:
            raise RuntimeError("litellm failure")
        return self.response

    def supports_reasoning(self, _model):
        return False

    def supports_function_calling(self, _model):
        if self.supports_function_calling_error:
            raise RuntimeError("helper unavailable")
        return self.supports_function_calling_value


class _BlockingFakeLitellm(_FakeLitellm):
    def __init__(self):
        super().__init__()
        self._all_entered = threading.Event()
        self._release = threading.Event()
        self._condition = threading.Condition()
        self._entered_count = 0

    def completion(self, **kwargs):
        payload = dict(kwargs)
        payload["_drop_params"] = self.drop_params
        payload["_verbose"] = self.verbose
        with self._condition:
            self.calls.append(payload)
            self._entered_count += 1
            if self._entered_count == 2:
                self._all_entered.set()

        if not self._release.wait(timeout=2.0):
            raise TimeoutError("test did not release fake LiteLLM completion")
        return self.response

    def wait_until_two_calls_entered(self) -> bool:
        return self._all_entered.wait(timeout=1.0)

    def release_all(self) -> None:
        self._release.set()


class _AsyncStreamingFakeLitellm(_FakeLitellm):
    def __init__(self, *, stream_response=None):
        super().__init__()
        self.stream_response = stream_response
        self.used_async_stream = False

    async def acompletion(self, **kwargs):
        self.async_calls.append(dict(kwargs))
        if kwargs.get("stream"):
            self.used_async_stream = True
            if self.stream_response is not None:
                return self.stream_response

            async def _chunks():
                yield {"choices": [{"delta": {"content": "hello "}}]}
                yield {"choices": [{"delta": {"content": "async"}, "finish_reason": "stop"}]}

            return _chunks()
        return await super().acompletion(**kwargs)


class _AsyncRouter:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []
        self.async_calls = []

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "router-sync"}}]}

    async def acompletion(self, **kwargs):
        self.async_calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "router-async"}}]}


class _LegacyRouterWithoutAsync:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "router-sync"}}]}


class _RouterFakeLitellm(_FakeLitellm):
    def __init__(self, router_cls):
        super().__init__()
        self.Router = router_cls


class _RetryAsyncFakeLitellm(_FakeLitellm):
    def __init__(self, *, failures: list[Exception], response: object | None = None):
        super().__init__(response=response or {"choices": [{"message": {"content": "retry-ok"}}]})
        self.failures = list(failures)

    async def acompletion(self, **kwargs):
        self.async_calls.append(dict(kwargs))
        if self.failures:
            raise self.failures.pop(0)
        return self.response


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

    def test_litellm_backend_keeps_drop_params_request_scoped_and_does_not_mutate_verbose(self):
        fake_litellm = _FakeLitellm()
        fake_litellm.drop_params = "persist-drop"
        fake_litellm.verbose = "persist-verbose"
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend({"model": "gpt-4o-mini", "verbose": True})
            backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(fake_litellm.calls[0]["drop_params"], True)
        self.assertEqual(fake_litellm.calls[0]["_drop_params"], "persist-drop")
        self.assertEqual(fake_litellm.calls[0]["_verbose"], "persist-verbose")
        self.assertEqual(fake_litellm.drop_params, "persist-drop")
        self.assertEqual(fake_litellm.verbose, "persist-verbose")

    def test_litellm_direct_calls_are_not_serialized_by_backend_lock(self):
        fake_litellm = _BlockingFakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend({"model": "gpt-4o-mini", "retry_sleep": 0.0})
            with ThreadPoolExecutor(max_workers=2) as pool:
                futures = [
                    pool.submit(backend.generate, {"messages": [{"role": "user", "content": "ping-a"}]}),
                    pool.submit(backend.generate, {"messages": [{"role": "user", "content": "ping-b"}]}),
                ]
                try:
                    self.assertTrue(fake_litellm.wait_until_two_calls_entered())
                finally:
                    fake_litellm.release_all()
                results = [future.result(timeout=2.0) for future in futures]

        self.assertEqual([result["answer"] for result in results], ["pong-lite", "pong-lite"])
        self.assertEqual(len(fake_litellm.calls), 2)
        self.assertTrue(all(call["drop_params"] is True for call in fake_litellm.calls))

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

    def test_litellm_request_extra_body_uses_request_scoped_drop_params(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/demo-model",
                    "api_key": "dummy",
                    "generation_parameters": {"max_new_tokens": 16},
                    "litellm_request": {
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
                    },
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["extra_body"], {"chat_template_kwargs": {"enable_thinking": False}})
        self.assertIs(call["drop_params"], True)
        self.assertNotIn("response_format", call)

    def test_litellm_request_response_format_is_forwarded_only_when_configured(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            default_backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            configured_backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                    "litellm_request": {"response_format": {"type": "json_object"}},
                }
            )
            default_backend.generate({"messages": [{"role": "user", "content": "ping"}]})
            configured_backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertNotIn("response_format", fake_litellm.calls[0])
        self.assertEqual(fake_litellm.calls[1]["response_format"], {"type": "json_object"})

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
        self.assertEqual(len(fake_litellm.async_calls), 2)
        self.assertEqual(fake_litellm.calls, [])

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

    def test_openai_multimodal_messages_keep_audio_url_blocks(self):
        fake_litellm = _FakeLitellm()
        audio_url = "data:audio/wav;base64,QUJDRA=="
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
                                {"type": "text", "text": "Transcribe this."},
                                {"type": "audio_url", "audio_url": {"url": audio_url}},
                            ],
                        }
                    ]
                }
            )
            result = backend.generate(prepared)

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["messages"][0]["content"][1], {"type": "audio_url", "audio_url": {"url": audio_url}})

    def test_language_model_only_rejects_multimodal_before_litellm_call(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/text-only",
                    "api_key": "dummy",
                    "generation_parameters": {"max_new_tokens": 16},
                    "vllm": {"topology": {"language_model_only": True}},
                }
            )
            with self.assertRaisesRegex(ValueError, "language_model_only"):
                backend.generate(
                    {
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                                ],
                            }
                        ]
                    }
                )

        self.assertEqual(fake_litellm.calls, [])

    def test_language_model_only_allows_text_only_list_content(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/text-only",
                    "api_key": "dummy",
                    "generation_parameters": {"max_new_tokens": 16},
                    "vllm": {"topology": {"language_model_only": True}},
                }
            )
            result = backend.generate(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": "hello"},
                            ],
                        }
                    ]
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        self.assertEqual(len(fake_litellm.calls), 1)
        self.assertEqual(fake_litellm.calls[0]["messages"][0]["content"], [{"type": "text", "text": "hello"}])

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

    def test_mock_fields_are_mapped_with_deprecation_warning(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}), mock.patch(
            "gage_eval.role.model.backends.litellm_backend.logger.warning"
        ) as warning:
            backend = LiteLLMBackend(
                {
                    "mock_api_base": "http://127.0.0.1:8000/v1",
                    "mock_api_key": "mock-key",
                    "mock_model": "hosted_vllm/mock-qwen",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertEqual(call["model"], "hosted_vllm/mock-qwen")
        self.assertEqual(call["api_base"], "http://127.0.0.1:8000/v1")
        self.assertEqual(call["api_key"], "mock-key")
        warning_messages = [str(args[0]) for args, _kwargs in warning.call_args_list]
        self.assertTrue(any("mock_api_base" in message and "deprecated" in message for message in warning_messages))

    def test_thinking_mode_disabled_uses_vllm_extra_body(self):
        """Thinking mode 'disabled' should use vLLM chat template kwargs, not a top-level kwarg."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/qwen3-32b",
                    "thinking_mode": "disabled",
                    "generation_parameters": {"max_new_tokens": 16},
                    "vllm": {"reasoning_parser": "qwen3"},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "solve 2+2"}], "sampling_params": {"max_new_tokens": 16}}
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertNotIn("enable_thinking", call)
        self.assertFalse(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_thinking_mode_enabled_uses_vllm_extra_body(self):
        """Thinking mode 'enabled' should use vLLM chat template kwargs, not a top-level kwarg."""
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/qwen3-32b",
                    "thinking_mode": "enabled",
                    "generation_parameters": {"max_new_tokens": 16},
                    "vllm": {"reasoning_parser": "qwen3"},
                }
            )
            result = backend.generate(
                {"messages": [{"role": "user", "content": "solve 2+2"}], "sampling_params": {"max_new_tokens": 16}}
            )

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertNotIn("enable_thinking", call)
        self.assertTrue(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_top_level_thinking_mode_overrides_generation_parameters_after_prepare_inputs(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/qwen3-32b",
                    "thinking_mode": "disabled",
                    "generation_parameters": {
                        "max_new_tokens": 16,
                        "thinking_mode": "enabled",
                    },
                    "vllm": {"reasoning_parser": "qwen3"},
                }
            )
            prepared = backend.prepare_inputs({"messages": [{"role": "user", "content": "solve 2+2"}]})
            result = backend.generate(prepared)

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertNotIn("enable_thinking", call)
        self.assertFalse(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

    def test_payload_thinking_mode_overrides_top_level_after_prepare_inputs(self):
        fake_litellm = _FakeLitellm()
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/qwen3-32b",
                    "thinking_mode": "disabled",
                    "generation_parameters": {
                        "max_new_tokens": 16,
                        "thinking_mode": "enabled",
                    },
                    "vllm": {"reasoning_parser": "qwen3"},
                }
            )
            prepared = backend.prepare_inputs(
                {
                    "messages": [{"role": "user", "content": "solve 2+2"}],
                    "sampling_params": {"thinking_mode": "enabled"},
                }
            )
            result = backend.generate(prepared)

        self.assertEqual(result["answer"], "pong-lite")
        call = fake_litellm.calls[0]
        self.assertNotIn("enable_thinking", call)
        self.assertTrue(call["extra_body"]["chat_template_kwargs"]["enable_thinking"])

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
            backend.generate(
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
        self.assertEqual(result["finish_reason"], "stop")
        self.assertEqual(result["usage"], {"prompt_tokens": 3, "completion_tokens": 5, "total_tokens": 8})
        self.assertEqual(result["tool_calls"][0]["id"], "tool-1")
        self.assertTrue(result["metadata"]["observation_summary"]["has_tool_calls"])
        message, payload = mock_logger_info.call_args_list[-1].args
        self.assertEqual(message, "LiteLLM response summary: {}")
        self.assertNotIn("super-secret-answer", payload)
        self.assertNotIn("tool-1", payload)
        self.assertIn("\"answer_chars\": 19", payload)
        self.assertIn("\"has_tool_calls\": true", payload)
        self.assertIn("\"finish_reason\": \"stop\"", payload)

    def test_tools_fail_fast_when_litellm_reports_function_calling_not_supported(self):
        fake_litellm = _FakeLitellm(supports_function_calling=False)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            with self.assertRaisesRegex(ValueError, "tool_calling_not_supported"):
                backend.generate(
                    {
                        "messages": [{"role": "user", "content": "ping"}],
                        "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                    }
                )

        self.assertEqual(fake_litellm.calls, [])

    def test_hosted_vllm_tools_with_declared_parser_ignore_false_function_calling_helper(self):
        fake_litellm = _FakeLitellm(supports_function_calling=False)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "hosted_vllm/demo-model",
                    "api_key": "dummy",
                    "generation_parameters": {"max_new_tokens": 16},
                    "tool_calling": {
                        "expected_server_flags": {
                            "enable_auto_tool_choice": True,
                            "tool_call_parser": "qwen3_xml",
                        }
                    },
                }
            )
            result = backend.generate(
                {
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                    "tool_choice": "auto",
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        self.assertEqual(fake_litellm.calls[0]["tool_choice"], "auto")
        self.assertEqual(fake_litellm.calls[0]["tools"][0]["function"]["name"], "lookup")

    def test_tools_call_litellm_when_function_calling_is_supported(self):
        fake_litellm = _FakeLitellm(supports_function_calling=True)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate(
                {
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        self.assertEqual(fake_litellm.calls[0]["tools"][0]["function"]["name"], "lookup")

    def test_tools_are_allowed_when_function_calling_helper_fails_unknown(self):
        fake_litellm = _FakeLitellm(supports_function_calling_error=True)
        with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
            backend = LiteLLMBackend(
                {
                    "model": "gpt-4o-mini",
                    "generation_parameters": {"max_new_tokens": 16},
                }
            )
            result = backend.generate(
                {
                    "messages": [{"role": "user", "content": "ping"}],
                    "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
                }
            )

        self.assertEqual(result["answer"], "pong-lite")
        self.assertEqual(fake_litellm.calls[0]["tools"][0]["function"]["name"], "lookup")


@pytest.mark.asyncio
async def test_async_invoke_uses_litellm_acompletion_without_sync_completion():
    fake_litellm = _FakeLitellm(response={"choices": [{"message": {"content": "async-ok"}}]})
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend({"model": "gpt-4o-mini", "generation_parameters": {"max_new_tokens": 16}})
        result = await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "async-ok"
    assert len(fake_litellm.async_calls) == 1
    assert fake_litellm.calls == []
    assert fake_litellm.async_calls[0]["stream"] is False


@pytest.mark.asyncio
async def test_async_streaming_uses_async_chunks_and_stream_normalizer():
    fake_litellm = _AsyncStreamingFakeLitellm()
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "gpt-4o-mini",
                "streaming": True,
                "generation_parameters": {"max_new_tokens": 16},
            }
        )
        result = await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "hello async"
    assert result["finish_reason"] == "stop"
    assert fake_litellm.used_async_stream is True
    assert len(fake_litellm.async_calls) == 1
    assert fake_litellm.calls == []


@pytest.mark.asyncio
async def test_async_streaming_accepts_completion_like_mapping_without_losing_answer():
    fake_litellm = _AsyncStreamingFakeLitellm(
        stream_response={"choices": [{"message": {"content": "mapping-stream-ok"}, "finish_reason": "stop"}]}
    )
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "gpt-4o-mini",
                "streaming": True,
                "generation_parameters": {"max_new_tokens": 16},
            }
        )
        result = await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "mapping-stream-ok"
    assert result["finish_reason"] == "stop"
    assert len(fake_litellm.async_calls) == 1
    assert fake_litellm.calls == []


@pytest.mark.asyncio
async def test_async_router_mode_calls_router_acompletion():
    fake_litellm = _RouterFakeLitellm(_AsyncRouter)
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "qwen3-group",
                "model_list": [
                    {
                        "model_name": "qwen3-group",
                        "litellm_params": {
                            "model": "hosted_vllm/qwen3",
                            "api_base": "http://127.0.0.1:8000/v1",
                            "api_key": "dummy",
                        },
                    }
                ],
                "generation_parameters": {"max_new_tokens": 16},
            }
        )
        result = await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "router-async"
    assert fake_litellm.calls == []
    assert backend._router.calls == []
    assert len(backend._router.async_calls) == 1
    assert backend._router.async_calls[0]["model"] == "qwen3-group"
    assert "api_base" not in backend._router.async_calls[0]


@pytest.mark.asyncio
async def test_async_router_mode_requires_router_acompletion():
    fake_litellm = _RouterFakeLitellm(_LegacyRouterWithoutAsync)
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "qwen3-group",
                "model_list": [
                    {
                        "model_name": "qwen3-group",
                        "litellm_params": {
                            "model": "hosted_vllm/qwen3",
                            "api_base": "http://127.0.0.1:8000/v1",
                            "api_key": "dummy",
                        },
                    }
                ],
                "generation_parameters": {"max_new_tokens": 16},
            }
        )
        with pytest.raises(RuntimeError, match="acompletion|async"):
            await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})


def test_wrap_backend_sync_invoke_uses_litellm_completion_not_acompletion():
    fake_litellm = _FakeLitellm(response={"choices": [{"message": {"content": "sync-ok"}}]})
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend({"model": "gpt-4o-mini", "generation_parameters": {"max_new_tokens": 16}})
        wrapped = wrap_backend(backend)
        result = wrapped.invoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "sync-ok"
    assert len(fake_litellm.calls) == 1
    assert fake_litellm.async_calls == []


def test_dut_model_adapter_sync_invoke_uses_litellm_completion_not_acompletion():
    fake_litellm = _FakeLitellm(response={"choices": [{"message": {"content": "adapter-sync-ok"}}]})
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend({"model": "gpt-4o-mini", "generation_parameters": {"max_new_tokens": 16}})
        adapter = DUTModelAdapter(
            adapter_id="dut",
            role_type="dut_model",
            backend=backend,
            capabilities=(),
        )
        result = adapter.invoke({"sample": {"id": "sample-1", "text": "hello"}}, RoleAdapterState())

    assert result["answer"] == "adapter-sync-ok"
    assert len(fake_litellm.calls) == 1
    assert fake_litellm.async_calls == []


@pytest.mark.asyncio
async def test_async_litellm_retries_until_success():
    fake_litellm = _RetryAsyncFakeLitellm(failures=[RuntimeError("temporary-1"), RuntimeError("temporary-2")])
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "gpt-4o-mini",
                "generation_parameters": {"max_new_tokens": 16},
                "max_retries": 3,
                "retry_sleep": 0.0,
            }
        )
        result = await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "retry-ok"
    assert len(fake_litellm.async_calls) == 3
    assert fake_litellm.calls == []


@pytest.mark.asyncio
async def test_async_litellm_non_retryable_shutdown_error_is_not_retried():
    fake_litellm = _RetryAsyncFakeLitellm(
        failures=[RuntimeError("cannot schedule new futures after shutdown")],
    )
    with mock.patch.dict(sys.modules, {"litellm": fake_litellm}):
        backend = LiteLLMBackend(
            {
                "model": "gpt-4o-mini",
                "generation_parameters": {"max_new_tokens": 16},
                "max_retries": 3,
                "retry_sleep": 0.0,
            }
        )
        with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
            await backend.ainvoke({"messages": [{"role": "user", "content": "ping"}]})

    assert len(fake_litellm.async_calls) == 1
    assert fake_litellm.calls == []


if __name__ == "__main__":
    unittest.main()
