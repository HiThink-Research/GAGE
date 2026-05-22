from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from gage_eval.role.model.backends import wrap_backend
from gage_eval.role.model.backends.litellm.errors import DEPENDENCY_UNAVAILABLE, LiteLLMBackendError
from gage_eval.role.model.backends.litellm.model_server_lifecycle import ModelServerLifecycle
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend

pytestmark = [pytest.mark.compat, pytest.mark.fast]


class _ProviderStatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


class _FakeRouter:
    def __init__(
        self,
        *,
        response: Any | None = None,
        stream_chunks: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> None:
        self.init_kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        self.async_calls: list[dict[str, Any]] = []
        self.response = response or {"choices": [{"message": {"content": "router-ok"}}]}
        self.stream_chunks = stream_chunks or _stream_chunks("router")

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return list(self.stream_chunks)
        return self.response

    async def acompletion(self, **kwargs: Any) -> Any:
        self.async_calls.append(dict(kwargs))
        if kwargs.get("stream"):
            return _async_chunks(self.stream_chunks)
        return self.response


class _FakeLiteLLM(types.SimpleNamespace):
    def __init__(
        self,
        *,
        response: Any | None = None,
        stream_chunks: list[dict[str, Any]] | None = None,
        async_stream_chunks: list[dict[str, Any]] | None = None,
        error: Exception | None = None,
        supports_reasoning: bool = False,
        supports_function_calling: bool | None = True,
    ) -> None:
        self.calls: list[dict[str, Any]] = []
        self.async_calls: list[dict[str, Any]] = []
        self.routers: list[_FakeRouter] = []
        self.response = response or {"choices": [{"message": {"content": "direct-ok"}}]}
        self.stream_chunks = stream_chunks or _stream_chunks("sync")
        self.async_stream_chunks = async_stream_chunks or _stream_chunks("async")
        self.error = error
        self.supports_reasoning_value = supports_reasoning
        self.supports_function_calling_value = supports_function_calling
        self.drop_params = False
        self.verbose = False

        def _router_factory(**kwargs: Any) -> _FakeRouter:
            router = _FakeRouter(response={"choices": [{"message": {"content": "router-ok"}}]}, **kwargs)
            self.routers.append(router)
            return router

        super().__init__(Router=_router_factory)

    def completion(self, **kwargs: Any) -> Any:
        self.calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return list(self.stream_chunks)
        return self.response

    async def acompletion(self, **kwargs: Any) -> Any:
        self.async_calls.append(dict(kwargs))
        if self.error is not None:
            raise self.error
        if kwargs.get("stream"):
            return _async_chunks(self.async_stream_chunks)
        return self.response

    def supports_reasoning(self, _model: str) -> bool:
        return self.supports_reasoning_value

    def supports_function_calling(self, _model: str) -> bool | None:
        return self.supports_function_calling_value


class _FakeProcess:
    def __init__(self, returncode: int | None = None) -> None:
        self.returncode = returncode

    def poll(self) -> int | None:
        return self.returncode


class _FakeClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


async def _async_chunks(chunks: list[dict[str, Any]]) -> Any:
    for chunk in chunks:
        yield chunk


def _stream_chunks(label: str) -> list[dict[str, Any]]:
    return [
        {"choices": [{"delta": {"content": f"{label}-"}}]},
        {
            "choices": [{"delta": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
        },
    ]


def _install_fake_litellm(monkeypatch: pytest.MonkeyPatch, fake: _FakeLiteLLM) -> _FakeLiteLLM:
    monkeypatch.setitem(sys.modules, "litellm", fake)
    return fake


def test_direct_hosted_vllm_completion_uses_direct_transport_and_gage_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(monkeypatch, _FakeLiteLLM())

    backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/Qwen/Qwen3.6-35B-A3B",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16, "temperature": 0.0},
            "retry_sleep": 0.0,
        }
    )
    result = backend.invoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "direct-ok"
    assert len(fake_litellm.calls) == 1
    call = fake_litellm.calls[0]
    assert call["model"] == "hosted_vllm/Qwen/Qwen3.6-35B-A3B"
    assert call["api_base"] == "http://127.0.0.1:8000/v1"
    assert call["base_url"] == "http://127.0.0.1:8000/v1"
    assert call["api_key"] == "dummy"
    assert call["custom_llm_provider"] == "hosted_vllm"
    assert call["max_tokens"] == 16
    summary = result["metadata"]["observation_summary"]
    assert summary["route_mode"] == "direct"
    assert summary["retry_owner"] == "gage"


def test_model_list_uses_litellm_router_and_strips_direct_transport_fields(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(monkeypatch, _FakeLiteLLM())

    backend = LiteLLMBackend(
        {
            "model": "qwen3-pool",
            "api_base": "http://wrong-top-level.example/v1",
            "api_key": "wrong-top-level-key",
            "custom_llm_provider": "openai",
            "extra_headers": {"X-Wrong": "top-level"},
            "max_retries": 6,
            "model_list": [
                {
                    "model_name": "qwen3-pool",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3-a",
                        "api_base": "http://127.0.0.1:8000/v1",
                        "api_key": "deployment-key-a",
                    },
                }
            ],
            "router_settings": {"num_retries": 2},
            "generation_parameters": {"max_new_tokens": 32, "temperature": 0.2},
        }
    )
    result = backend.invoke({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "router-ok"
    assert fake_litellm.calls == []
    assert backend._max_retries == 1
    router = fake_litellm.routers[0]
    assert router.init_kwargs["model_list"][0]["litellm_params"]["api_base"] == "http://127.0.0.1:8000/v1"
    assert router.init_kwargs["model_list"][0]["litellm_params"]["api_key"] == "deployment-key-a"
    assert router.init_kwargs["num_retries"] == 2

    call = router.calls[0]
    assert call["model"] == "qwen3-pool"
    assert call["messages"] == [{"role": "user", "content": "ping"}]
    assert call["max_tokens"] == 32
    assert call["temperature"] == 0.2
    assert "api_base" not in call
    assert "base_url" not in call
    assert "api_key" not in call
    assert "custom_llm_provider" not in call
    assert "headers" not in call
    assert result["metadata"]["observation_summary"]["retry_owner"] == "litellm_router"


def test_external_gateway_uses_litellm_api_key_and_gateway_retry_owner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(monkeypatch, _FakeLiteLLM())
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-key")

    backend = LiteLLMBackend(
        {
            "model": "openai/Qwen/Qwen3.6-35B-A3B",
            "api_base": "http://127.0.0.1:4000/v1",
            "max_retries": 6,
            "vllm": {"topology": {"kind": "external_gateway"}},
            "generation_parameters": {"max_new_tokens": 16},
            "retry_sleep": 0.0,
        }
    )
    result = backend.invoke({"messages": [{"role": "user", "content": "ping"}]})

    call = fake_litellm.calls[0]
    assert call["api_base"] == "http://127.0.0.1:4000/v1"
    assert call["api_key"] == "gateway-key"
    assert call["api_key"] != "openai-key"
    assert backend._max_retries == 1
    summary = result["metadata"]["observation_summary"]
    assert summary["retry_owner"] == "litellm_gateway"
    assert summary["topology"]["kind"] == "external_gateway"


def test_thinking_controls_map_to_extra_body_and_gate_reasoning_token_expansion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(monkeypatch, _FakeLiteLLM(supports_reasoning=True))

    disabled_backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen3",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16, "thinking_mode": "disabled"},
            "vllm": {"reasoning_parser": "qwen3"},
        }
    )
    disabled_backend.invoke({"messages": [{"role": "user", "content": "no thinking"}]})

    enabled_backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen3",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16, "thinking_mode": "enabled"},
            "vllm": {"reasoning_parser": "qwen3"},
        }
    )
    enabled_backend.invoke({"messages": [{"role": "user", "content": "think"}]})

    disabled_call, enabled_call = fake_litellm.calls
    assert disabled_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert disabled_call["max_tokens"] == 16
    assert enabled_call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is True
    assert enabled_call["max_tokens"] == 160


def test_disabled_thinking_records_mismatch_when_response_contains_reasoning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(
        monkeypatch,
        _FakeLiteLLM(response={"choices": [{"message": {"content": "<think>hidden</think>\nvisible"}}]}),
    )

    backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen3",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16, "thinking_mode": "disabled"},
            "thinking_policy": {"on_mismatch": "warn_and_continue"},
            "vllm": {"reasoning_parser": "qwen3"},
        }
    )
    result = backend.invoke({"messages": [{"role": "user", "content": "answer only"}]})

    assert len(fake_litellm.calls) == 1
    assert result["answer"] == "visible"
    assert result["reasoning_content"] == "hidden"
    assert result["metadata"]["thinking_effective"] == "enabled"
    assert result["metadata"]["thinking_mismatch"] == {
        "expected": "disabled",
        "observed": "enabled",
        "error_type": "response_mismatch",
        "reasoning_source": "think_tags",
    }


@pytest.mark.asyncio
async def test_sync_and_async_streaming_normalize_answer_usage_and_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sync_fake = _install_fake_litellm(monkeypatch, _FakeLiteLLM(stream_chunks=_stream_chunks("sync")))
    sync_backend = LiteLLMBackend(
        {
            "model": "gpt-4o-mini",
            "api_key": "openai-key",
            "streaming": True,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )
    sync_result = sync_backend.invoke({"messages": [{"role": "user", "content": "stream"}]})

    assert sync_result["answer"] == "sync-ok"
    assert sync_result["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert sync_result["finish_reason"] == "stop"
    assert sync_result["metadata"]["observation_summary"]["stream"] is True
    assert sync_fake.calls[0]["stream"] is True

    async_fake = _install_fake_litellm(monkeypatch, _FakeLiteLLM(async_stream_chunks=_stream_chunks("async")))
    async_backend = LiteLLMBackend(
        {
            "model": "gpt-4o-mini",
            "api_key": "openai-key",
            "streaming": True,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )
    async_result = await async_backend.ainvoke({"messages": [{"role": "user", "content": "stream"}]})

    assert async_result["answer"] == "async-ok"
    assert async_result["usage"] == {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5}
    assert async_result["metadata"]["observation_summary"]["stream"] is True
    assert len(async_fake.async_calls) == 1
    assert async_fake.calls == []
    assert async_fake.async_calls[0]["stream"] is True


def test_multimodal_payload_errors_fail_fast_before_litellm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(monkeypatch, _FakeLiteLLM())

    text_only_backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/text-only",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16},
            "vllm": {"topology": {"language_model_only": True}},
        }
    )
    with pytest.raises(ValueError, match="language_model_only.*error_type=unsupported_capability"):
        text_only_backend.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/a.png"}}],
                    }
                ]
            }
        )

    invalid_payload_backend = LiteLLMBackend(
        {
            "model": "gpt-4o-audio-preview",
            "api_key": "openai-key",
            "generation_parameters": {"max_new_tokens": 16},
        }
    )
    with pytest.raises(ValueError, match="invalid_media_payload"):
        invalid_payload_backend.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"type": "input_audio", "input_audio": {"data": "%%%!", "format": "wav"}}],
                    }
                ]
            }
        )

    assert fake_litellm.calls == []


def test_tool_calling_request_and_response_contract_preserves_invalid_arguments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = _install_fake_litellm(
        monkeypatch,
        _FakeLiteLLM(
            supports_function_calling=True,
            response={
                "choices": [
                    {
                        "finish_reason": "tool_calls",
                        "message": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "call_good",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{\"city\":\"Shanghai\"}"},
                                },
                                {
                                    "id": "call_bad",
                                    "type": "function",
                                    "function": {"name": "lookup", "arguments": "{bad-json"},
                                },
                            ],
                        },
                    }
                ]
            },
        ),
    )

    backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen3-tools",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16},
            "tool_calling": {
                "enabled": True,
                "auto_tool_choice": True,
                "parallel_tool_calls": True,
                "expected_server_flags": {
                    "enable_auto_tool_choice": True,
                    "tool_call_parser": "qwen3_xml",
                },
            },
        }
    )
    result = backend.invoke(
        {
            "messages": [{"role": "user", "content": "weather"}],
            "tools": [
                {
                    "name": "lookup",
                    "description": "Lookup weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ],
        }
    )

    call = fake_litellm.calls[0]
    assert call["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "lookup",
                "description": "Lookup weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
    ]
    assert call["tool_choice"] == "auto"
    assert call["parallel_tool_calls"] is True
    assert result["finish_reason"] == "tool_calls"
    assert result["tool_calls"][0]["arguments"] == {"city": "Shanghai"}
    assert result["tool_calls"][0]["raw_arguments"] == "{\"city\":\"Shanghai\"}"
    assert result["tool_calls"][1]["arguments"] is None
    assert result["tool_calls"][1]["raw_arguments"] == "{bad-json"
    assert result["tool_calls"][1]["error"] == "invalid_tool_arguments"
    assert result["metadata"]["observation_summary"]["tool_count"] == 1
    assert result["metadata"]["observation_summary"]["tool_call_count"] == 2


def test_error_classification_contracts_are_normalized_without_retrying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for status_code in (400, 422):
        fake_litellm = _install_fake_litellm(
            monkeypatch,
            _FakeLiteLLM(error=_ProviderStatusError("provider rejected request", status_code)),
        )
        backend = LiteLLMBackend(
            {
                "model": "gpt-4o-mini",
                "api_key": "openai-key",
                "max_retries": 4,
                "retry_sleep": 0.0,
                "generation_parameters": {"max_new_tokens": 16},
            }
        )

        result = wrap_backend(backend).invoke({"messages": [{"role": "user", "content": "bad request"}]})

        assert len(fake_litellm.calls) == 1
        assert result["status"] == status_code
        assert result["backend"] == "LiteLLMBackend"

    dependency_fake = _install_fake_litellm(monkeypatch, _FakeLiteLLM(error=RuntimeError("connection refused")))
    dependency_backend = LiteLLMBackend(
        {
            "model": "gpt-4o-mini",
            "api_key": "openai-key",
            "max_retries": 2,
            "retry_sleep": 0.0,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )
    dependency_result = wrap_backend(dependency_backend).invoke(
        {"messages": [{"role": "user", "content": "dependency"}]}
    )

    assert len(dependency_fake.calls) == 2
    assert dependency_result["error_type"] == DEPENDENCY_UNAVAILABLE
    assert dependency_result["backend"] == "LiteLLMBackend"

    unsupported_fake = _install_fake_litellm(
        monkeypatch,
        _FakeLiteLLM(supports_function_calling=False),
    )
    unsupported_backend = LiteLLMBackend(
        {
            "model": "gpt-4o-mini",
            "api_key": "openai-key",
            "generation_parameters": {"max_new_tokens": 16},
        }
    )
    unsupported_result = wrap_backend(unsupported_backend).invoke(
        {
            "messages": [{"role": "user", "content": "tools"}],
            "tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}],
        }
    )

    assert unsupported_fake.calls == []
    assert unsupported_result["error_type"] == "unsupported_capability"
    assert unsupported_result["backend"] == "LiteLLMBackend"


def test_model_server_lifecycle_failures_are_dependency_unavailable_without_real_io() -> None:
    timeout_clock = _FakeClock()
    timeout_lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: _FakeProcess(),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
        sleep=timeout_clock.sleep,
        monotonic=timeout_clock.monotonic,
    )
    config = {
        "enabled": True,
        "startup_command": "vllm serve /models/qwen --port 8000",
        "health": {
            "urls": ["http://127.0.0.1:8000/health"],
            "timeout_seconds": 0.25,
            "interval_seconds": 0.1,
        },
    }

    with pytest.raises(LiteLLMBackendError) as timeout_error:
        timeout_lifecycle.ensure_ready(config)

    assert timeout_error.value.error_type == DEPENDENCY_UNAVAILABLE
    assert "timed out" in str(timeout_error.value).lower()
    assert timeout_clock.sleeps

    popen_lifecycle = ModelServerLifecycle(
        popen=lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("bash unavailable")),
        urlopen=lambda *_args, **_kwargs: (_ for _ in ()).throw(TimeoutError("not ready")),
    )

    with pytest.raises(LiteLLMBackendError) as popen_error:
        popen_lifecycle.ensure_ready(config)

    assert popen_error.value.error_type == DEPENDENCY_UNAVAILABLE
    assert "bash unavailable" in str(popen_error.value)
