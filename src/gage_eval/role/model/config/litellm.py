"""LiteLLM backend configuration."""

from __future__ import annotations

from importlib import import_module
from typing import Any, Dict, Literal, Optional

from pydantic import Field, field_validator

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


MIN_LITELLM_VERSION = "1.63.8"
MIN_VLLM_VERSION = "0.20.1"


def assert_litellm_capabilities(
    *,
    require_router: bool = False,
    require_async: bool = False,
    require_function_calling: bool = False,
    litellm_module: Any | None = None,
) -> None:
    """Fail fast when required LiteLLM SDK capabilities are unavailable."""

    if litellm_module is None:
        try:
            litellm_module = import_module("litellm")
        except ImportError as exc:
            raise RuntimeError(
                f"LiteLLM is not installed; install litellm>={MIN_LITELLM_VERSION} for the GAGE LiteLLM backend."
            ) from exc

    _require_callable(litellm_module, "completion", "LiteLLM completion")
    if require_router:
        _require_attribute(litellm_module, "Router", "LiteLLM Router")
    if require_async:
        _require_callable(litellm_module, "acompletion", "LiteLLM acompletion")
    if require_function_calling:
        _require_callable(litellm_module, "supports_function_calling", "LiteLLM supports_function_calling")


def _require_attribute(module: Any, attribute: str, capability_name: str) -> None:
    if getattr(module, attribute, None) is None:
        raise RuntimeError(
            f"{capability_name} is required by this LiteLLM backend feature. "
            f"Install litellm>={MIN_LITELLM_VERSION} and verify the active environment is not using an older LiteLLM package."
        )


def _require_callable(module: Any, attribute: str, capability_name: str) -> None:
    value = getattr(module, attribute, None)
    if not callable(value):
        raise RuntimeError(
            f"{capability_name} is required by this LiteLLM backend feature. "
            f"Install litellm>={MIN_LITELLM_VERSION} and verify the active environment is not using an older LiteLLM package."
        )


class LiteLLMRequestConfig(BackendConfigBase):
    """LiteLLM completion parameters passed through from backend config."""

    response_format: Dict[str, Any] | None = None
    max_completion_tokens: int | None = Field(default=None, ge=1)
    stream_options: Dict[str, Any] | None = None
    parallel_tool_calls: bool | None = None
    logit_bias: Dict[str, Any] | None = None
    user: str | None = None
    metadata: Dict[str, Any] | None = None
    fallbacks: list[Dict[str, Any]] | None = None
    context_window_fallback_dict: Dict[str, Any] | None = None
    extra_body: Dict[str, Any] = Field(default_factory=dict)
    drop_params: bool | None = None


class LiteLLMRouterDeploymentConfig(BackendConfigBase):
    """One LiteLLM Router deployment entry managed by the GAGE backend."""

    model_name: str
    litellm_params: Dict[str, Any] = Field(default_factory=dict)


class LiteLLMRouterSettingsConfig(BackendConfigBase):
    """LiteLLM Router settings preserved for later router construction."""

    routing_strategy: str | None = None
    allowed_fails: int | None = Field(default=None, ge=0)
    cooldown_time: float | None = Field(default=None, ge=0.0)
    num_retries: int | None = Field(default=None, ge=0)


class LiteLLMVLLMTopologyConfig(BackendConfigBase):
    """vLLM endpoint topology metadata used for reporting and checks."""

    language_model_only: bool | None = None
    kind: str = "external_endpoint"


class LiteLLMVLLMConfig(BackendConfigBase):
    """vLLM profile metadata that does not participate in request building."""

    topology: LiteLLMVLLMTopologyConfig = Field(default_factory=LiteLLMVLLMTopologyConfig)
    reasoning_parser: str | None = None


class LiteLLMThinkingPolicyConfig(BackendConfigBase):
    """Policy for handling model thinking support and mismatch cases."""

    capability: Literal["auto", "supported", "unsupported"] | None = None
    on_unsupported: Literal["warn_and_continue", "fail_fast", "ignore"] | None = None
    on_mismatch: Literal["warn_and_continue", "fail_fast", "ignore"] | None = None
    record_effective_state: bool = False

    @field_validator("capability", mode="before")
    @classmethod
    def _coerce_boolean_capability(cls, value: Any) -> Any:
        if value is True:
            return "supported"
        if value is False:
            return "unsupported"
        return value


class LiteLLMMultimodalPolicyConfig(BackendConfigBase):
    """Policy for preserving multimodal content blocks through LiteLLM."""

    preserve_blocks: bool = True
    audio_url_mapping: str | None = None
    default_audio_format: str | None = None
    max_media_bytes: int | None = Field(default=None, ge=1)
    fallback: Literal["fail", "preserve", "text"] = "fail"


class LiteLLMToolExpectedServerFlagsConfig(BackendConfigBase):
    """Expected server-side flags for vLLM tool-calling support."""

    enable_auto_tool_choice: bool | None = None
    tool_call_parser: str | None = None


class LiteLLMToolCallingConfig(BackendConfigBase):
    """Tool-calling capability expectations for LiteLLM-compatible servers."""

    enabled: bool = False
    tool_choice: Any | None = None
    parallel_tool_calls: bool | None = None
    auto_tool_choice: bool = False
    require_server_parser: bool = False
    expected_tool_parser: str | None = None
    expected_server_flags: LiteLLMToolExpectedServerFlagsConfig = Field(
        default_factory=LiteLLMToolExpectedServerFlagsConfig
    )


class ModelServerHealthConfig(BackendConfigBase):
    """Health-check settings for experimental local model server startup."""

    urls: list[str] = Field(default_factory=list)
    timeout_seconds: float | None = Field(default=None, ge=0.0)
    interval_seconds: float | None = Field(default=None, ge=0.0)


class ModelServerLifecycleConfig(BackendConfigBase):
    """Experimental command-mode local model server lifecycle config."""

    enabled: bool = False
    startup_command: str | None = None
    health: ModelServerHealthConfig = Field(default_factory=ModelServerHealthConfig)


class LiteLLMBackendConfig(BackendConfigBase):
    model: str = Field(default="gpt-4o-mini")
    provider: Optional[str] = None
    custom_llm_provider: Optional[str] = Field(default=None, description="显式传递给 litellm 的 provider 名称")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    streaming: bool = False
    embed_remote_images: bool = Field(
        default=False,
        description="Convert remote image_url entries to data URLs lazily before sending a request",
    )
    remote_image_timeout_s: float = Field(default=60.0, ge=0.0)
    drop_params: bool = True
    timeout: Optional[float] = Field(default=None, ge=0.0)
    max_retries: int = Field(default=6, ge=1)
    retry_sleep: float = Field(default=1.0, ge=0.0)
    retry_multiplier: float = Field(default=2.0, ge=1.0)
    max_model_length: Optional[int] = Field(default=None, ge=1)
    azure_api_version: Optional[str] = Field(default=None, description="Azure OpenAI API version，用于易用性默认填充")
    prefer_litellm_kimi: bool = False
    force_kimi_direct: bool = False
    verbose: bool = False
    mock_api_base: Optional[str] = Field(
        default=None, description="用于本地/自建 OpenAI 兼容服务的 base_url（无额度时模拟调用）"
    )
    mock_api_key: Optional[str] = Field(default=None, description="本地/自建服务的密钥（可留空走环境变量）")
    mock_model: Optional[str] = Field(default=None, description="模拟调用时使用的模型名；缺省沿用 model")
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
    route_mode: Literal["direct", "router"] = "direct"
    litellm_request: LiteLLMRequestConfig = Field(default_factory=LiteLLMRequestConfig)
    model_list: list[LiteLLMRouterDeploymentConfig] = Field(default_factory=list)
    router_settings: LiteLLMRouterSettingsConfig = Field(default_factory=LiteLLMRouterSettingsConfig)
    vllm: LiteLLMVLLMConfig = Field(default_factory=LiteLLMVLLMConfig)
    thinking_policy: LiteLLMThinkingPolicyConfig = Field(default_factory=LiteLLMThinkingPolicyConfig)
    multimodal: LiteLLMMultimodalPolicyConfig = Field(default_factory=LiteLLMMultimodalPolicyConfig)
    tool_calling: LiteLLMToolCallingConfig = Field(default_factory=LiteLLMToolCallingConfig)
    model_server: ModelServerLifecycleConfig = Field(default_factory=ModelServerLifecycleConfig)

    def resolved_route_mode(self) -> Literal["direct", "router"]:
        if self.route_mode == "router":
            return "router"
        if self._field_was_explicitly_set("route_mode"):
            return "direct"
        if self.model_list:
            return "router"
        return "direct"

    def _field_was_explicitly_set(self, field_name: str) -> bool:
        fields_set = getattr(self, "model_fields_set", None)
        if fields_set is None:
            fields_set = getattr(self, "__fields_set__", set())
        return field_name in fields_set
