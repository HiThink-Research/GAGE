from __future__ import annotations

import sys
import types

import pytest

from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend
from gage_eval.role.model.config.litellm import LiteLLMBackendConfig

pytestmark = pytest.mark.fast


class FakeRouter:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.calls = []

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "router-ok"}}]}


class FakeLiteLLM(types.SimpleNamespace):
    def __init__(self):
        super().__init__(Router=FakeRouter)
        self.calls = []
        self.verbose = False

    def completion(self, **kwargs):
        self.calls.append(dict(kwargs))
        return {"choices": [{"message": {"content": "direct-ok"}}]}

    def supports_reasoning(self, _model):
        return False


def test_router_factory_builds_router_from_gage_model_list() -> None:
    from gage_eval.role.model.backends.litellm.router_factory import LiteLLMRouterFactory

    fake_litellm = FakeLiteLLM()
    cfg = LiteLLMBackendConfig.model_validate(
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
                    "rpm": 60,
                }
            ],
            "router_settings": {
                "routing_strategy": "simple-shuffle",
                "allowed_fails": 3,
                "cooldown_time": 30,
                "num_retries": 2,
                "retry_after": 5,
            },
        }
    )

    router = LiteLLMRouterFactory(fake_litellm).build(cfg)

    assert isinstance(router, FakeRouter)
    assert router.init_kwargs["model_list"][0]["model_name"] == "qwen3-group"
    assert router.init_kwargs["model_list"][0]["litellm_params"]["api_base"] == "http://127.0.0.1:8000/v1"
    assert router.init_kwargs["model_list"][0]["rpm"] == 60
    assert router.init_kwargs["routing_strategy"] == "simple-shuffle"
    assert router.init_kwargs["allowed_fails"] == 3
    assert router.init_kwargs["cooldown_time"] == 30
    assert router.init_kwargs["num_retries"] == 2
    assert router.init_kwargs["retry_after"] == 5


def test_router_factory_direct_without_model_list_returns_none() -> None:
    from gage_eval.role.model.backends.litellm.router_factory import LiteLLMRouterFactory

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "openai/qwen3-gateway",
            "api_base": "http://127.0.0.1:4000/v1",
            "api_key": "dummy",
        }
    )

    assert LiteLLMRouterFactory(FakeLiteLLM()).build(cfg) is None


def test_router_factory_explicit_direct_with_model_list_returns_none() -> None:
    from gage_eval.role.model.backends.litellm.router_factory import LiteLLMRouterFactory

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "route_mode": "direct",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3",
                        "api_base": "http://127.0.0.1:8000/v1",
                        "api_key": "deployment-key",
                    },
                }
            ],
        }
    )

    assert LiteLLMRouterFactory(FakeLiteLLM()).build(cfg) is None


def test_router_factory_fails_fast_when_router_mode_has_empty_model_list() -> None:
    from gage_eval.role.model.backends.litellm.router_factory import LiteLLMRouterFactory

    cfg = LiteLLMBackendConfig.model_validate({"model": "qwen3-group", "route_mode": "router"})

    with pytest.raises(ValueError, match="model_list"):
        LiteLLMRouterFactory(FakeLiteLLM()).build(cfg)


def test_service_profile_warns_when_single_profile_applies_to_all_deployments() -> None:
    from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3-a",
                        "api_base": "http://127.0.0.1:8000/v1",
                    },
                },
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3-b",
                        "api_base": "http://127.0.0.1:8001/v1",
                    },
                },
            ],
            "vllm": {"reasoning_parser": "qwen3"},
        }
    )

    warnings = VLLMServiceProfile.from_config(cfg).validate_deployment_consistency()

    assert warnings == [
        "per-deployment vLLM metadata not provided; assuming declared profile applies to all deployments"
    ]


def test_service_profile_warns_when_top_level_language_model_only_lacks_deployment_metadata() -> None:
    from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-a"},
                },
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-b"},
                },
            ],
            "vllm": {"topology": {"language_model_only": True}},
        }
    )

    warnings = VLLMServiceProfile.from_config(cfg).validate_deployment_consistency()

    assert warnings == [
        "per-deployment vLLM metadata not provided; assuming declared profile applies to all deployments"
    ]


def test_service_profile_fails_on_inconsistent_per_deployment_reasoning_parser() -> None:
    from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-a"},
                    "vllm": {"reasoning_parser": "qwen3"},
                },
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-b"},
                    "vllm": {"reasoning_parser": "deepseek-r1"},
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="reasoning_parser"):
        VLLMServiceProfile.from_config(cfg).validate_deployment_consistency()


def test_service_profile_fails_on_inconsistent_per_deployment_chat_template_defaults() -> None:
    from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-a"},
                    "vllm": {"chat_template_defaults": {"enable_thinking": True}},
                },
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-b"},
                    "vllm": {"chat_template_defaults": {"enable_thinking": False}},
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="chat_template_defaults"):
        VLLMServiceProfile.from_config(cfg).validate_deployment_consistency()


def test_service_profile_fails_on_inconsistent_per_deployment_language_model_only() -> None:
    from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile

    cfg = LiteLLMBackendConfig.model_validate(
        {
            "model": "qwen3-group",
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-a"},
                    "vllm": {"topology": {"language_model_only": True}},
                },
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {"model": "hosted_vllm/qwen3-b"},
                    "language_model_only": False,
                },
            ],
        }
    )

    with pytest.raises(ValueError, match="language_model_only"):
        VLLMServiceProfile.from_config(cfg).validate_deployment_consistency()


def test_backend_router_mode_calls_router_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

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

    result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "router-ok"
    assert fake_litellm.calls == []
    assert isinstance(backend._router, FakeRouter)
    assert backend._router.calls[0]["model"] == "qwen3-group"
    assert backend._router.calls[0]["messages"] == [{"role": "user", "content": "ping"}]


def test_backend_router_completion_does_not_forward_direct_transport_kwargs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LiteLLMBackend(
        {
            "model": "qwen3-group",
            "api_base": "http://wrong-top-level.example/v1",
            "api_key": "wrong-top-level-key",
            "custom_llm_provider": "openai",
            "extra_headers": {"X-Wrong": "top-level"},
            "model_list": [
                {
                    "model_name": "qwen3-group",
                    "litellm_params": {
                        "model": "hosted_vllm/qwen3-a",
                        "api_base": "http://127.0.0.1:8000/v1",
                        "api_key": "deployment-key-a",
                    },
                }
            ],
            "generation_parameters": {"max_new_tokens": 16, "temperature": 0.2},
            "timeout": 30,
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "router-ok"
    call = backend._router.calls[0]
    assert call["model"] == "qwen3-group"
    assert call["messages"] == [{"role": "user", "content": "ping"}]
    assert call["max_tokens"] == 16
    assert call["temperature"] == 0.2
    assert call["timeout"] == 30
    assert "api_base" not in call
    assert "base_url" not in call
    assert "api_key" not in call
    assert "custom_llm_provider" not in call
    assert "api_type" not in call
    assert "api_version" not in call
    assert "headers" not in call


def test_backend_gateway_without_model_list_stays_direct(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_litellm = FakeLiteLLM()
    monkeypatch.setitem(sys.modules, "litellm", fake_litellm)

    backend = LiteLLMBackend(
        {
            "model": "openai/qwen3-gateway",
            "api_base": "http://127.0.0.1:4000/v1",
            "api_key": "dummy",
            "generation_parameters": {"max_new_tokens": 16},
        }
    )

    result = backend.generate({"messages": [{"role": "user", "content": "ping"}]})

    assert result["answer"] == "direct-ok"
    assert backend._router is None
    assert len(fake_litellm.calls) == 1
    assert fake_litellm.calls[0]["api_base"] == "http://127.0.0.1:4000/v1"
