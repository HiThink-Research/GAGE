from __future__ import annotations

import builtins
import sys

import pytest

from gage_eval.role.model.backends.litellm.errors import DEPENDENCY_UNAVAILABLE, LiteLLMBackendError
from gage_eval.role.model.backends.litellm.policies import ToolCallPolicy
from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


def _build_backend_for_retry_tests(*, max_retries: int) -> LiteLLMBackend:
    backend = LiteLLMBackend.__new__(LiteLLMBackend)
    backend._max_retries = max_retries
    backend._retry_sleep = 0.0
    backend._retry_multiplier = 1.0
    return backend


class _ProviderStatusError(RuntimeError):
    def __init__(self, message: str, status_code: int) -> None:
        super().__init__(message)
        self.status_code = status_code


@pytest.mark.fast
def test_call_with_retries_stops_for_shutdown_error() -> None:
    backend = _build_backend_for_retry_tests(max_retries=6)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("cannot schedule new futures after shutdown")

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        backend._call_with_retries(_call)

    assert attempts == 1


@pytest.mark.fast
def test_call_with_retries_retries_for_retryable_error() -> None:
    backend = _build_backend_for_retry_tests(max_retries=3)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary upstream error")

    with pytest.raises(RuntimeError, match="temporary upstream error"):
        backend._call_with_retries(_call)

    assert attempts == 3


@pytest.mark.fast
@pytest.mark.parametrize("status_code", [400, 401, 403, 404, 422])
def test_call_with_retries_does_not_retry_provider_client_errors(status_code: int) -> None:
    backend = _build_backend_for_retry_tests(max_retries=6)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise _ProviderStatusError("provider rejected request", status_code)

    with pytest.raises(_ProviderStatusError, match="provider rejected request"):
        backend._call_with_retries(_call)

    assert attempts == 1


@pytest.mark.fast
def test_call_with_retries_classifies_endpoint_unavailable_after_retry_budget() -> None:
    backend = _build_backend_for_retry_tests(max_retries=2)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("connection refused")

    with pytest.raises(LiteLLMBackendError) as exc_info:
        backend._call_with_retries(_call)

    assert attempts == 2
    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE


@pytest.mark.fast
def test_litellm_import_failure_is_dependency_unavailable(monkeypatch) -> None:
    real_import = builtins.__import__

    def _import(name, *args, **kwargs):
        if name == "litellm":
            raise ImportError("missing litellm")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import)

    with pytest.raises(LiteLLMBackendError) as exc_info:
        LiteLLMBackend({"model": "gpt-4o-mini"})

    assert exc_info.value.error_type == DEPENDENCY_UNAVAILABLE


@pytest.mark.fast
def test_router_mode_uses_router_retry_owner_without_backend_retry(monkeypatch) -> None:
    from tests.unit.role.model.test_litellm_router_factory import FakeLiteLLM

    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LiteLLMBackend(
        {
            "model": "qwen3-group",
            "max_retries": 6,
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3",
                        "api_base": "http://127.0.0.1:8000/v1",
                    },
                }
            ],
            "router_settings": {"num_retries": 2},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert backend._max_retries == 1
    assert result["metadata"]["observation_summary"]["retry_owner"] == "litellm_router"
    assert result["metadata"]["observation_summary"]["outer_retry_attempts"] == 1
    assert result["metadata"]["observation_summary"]["router_num_retries"] == 2


@pytest.mark.fast
def test_successful_retry_records_outer_retry_attempt_count(monkeypatch) -> None:
    from tests.unit.role.model.test_litellm_router_factory import FakeLiteLLM

    class FlakyLiteLLM(FakeLiteLLM):
        def completion(self, **kwargs):
            self.calls.append(dict(kwargs))
            if len(self.calls) == 1:
                raise RuntimeError("temporary upstream error")
            return {"choices": [{"message": {"content": "recovered"}}]}

    fake_litellm = FlakyLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LiteLLMBackend(
        {
            "model": "hosted_vllm/qwen3",
            "api_base": "http://127.0.0.1:8000/v1",
            "api_key": "dummy",
            "max_retries": 2,
            "retry_sleep": 0,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert result["answer"] == "recovered"
    assert len(fake_litellm.calls) == 2
    assert result["metadata"]["observation_summary"]["outer_retry_attempts"] == 2
    assert not hasattr(backend, "_last_outer_retry_attempts")


@pytest.mark.fast
def test_gateway_mode_uses_gateway_retry_owner_without_backend_retry(monkeypatch) -> None:
    from tests.unit.role.model.test_litellm_router_factory import FakeLiteLLM

    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.setenv("LITELLM_API_KEY", "gateway-key")

    backend = LiteLLMBackend(
        {
            "model": "openai/qwen3-group",
            "api_base": "http://127.0.0.1:4000/v1",
            "max_retries": 6,
            "vllm": {"topology": {"kind": "external_gateway"}},
            "generation_parameters": {"max_new_tokens": 16},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert backend._max_retries == 1
    assert result["metadata"]["observation_summary"]["retry_owner"] == "litellm_gateway"
    assert fake_litellm.calls[0]["api_key"] == "gateway-key"


@pytest.mark.fast
def test_default_local_gateway_config_uses_gateway_retry_owner_without_openai_key_leak(monkeypatch) -> None:
    from tests.unit.role.model.test_litellm_router_factory import FakeLiteLLM

    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    monkeypatch.delenv("LITELLM_API_KEY", raising=False)

    backend = LiteLLMBackend(
        {
            "model": "openai/qwen3-group",
            "api_base": "http://127.0.0.1:4000/v1",
            "max_retries": 6,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert backend._max_retries == 1
    assert result["metadata"]["observation_summary"]["retry_owner"] == "litellm_gateway"
    assert fake_litellm.calls[0]["api_key"] == "dummy"


@pytest.mark.fast
@pytest.mark.parametrize(
    "api_base",
    [
        "https://api-gateway.company.com/v1",
        "https://litellm-compatible.company.com/v1",
    ],
)
def test_gateway_like_direct_endpoint_keeps_gage_retry_owner_without_explicit_profile(
    monkeypatch,
    api_base: str,
) -> None:
    from tests.unit.role.model.test_litellm_router_factory import FakeLiteLLM

    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LiteLLMBackend(
        {
            "model": "openai/qwen3-group",
            "api_base": api_base,
            "max_retries": 6,
            "generation_parameters": {"max_new_tokens": 16},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "hello"}]})

    assert backend._max_retries == 6
    assert result["metadata"]["observation_summary"]["retry_owner"] == "gage"


@pytest.mark.fast
def test_tool_calling_unsupported_error_is_classified_as_unsupported_capability() -> None:
    policy = ToolCallPolicy()

    with pytest.raises(ValueError, match="error_type=unsupported_capability"):
        policy.apply(
            {},
            {"tools": [{"name": "lookup", "parameters": {"type": "object", "properties": {}}}]},
            supports_function_calling=False,
        )
