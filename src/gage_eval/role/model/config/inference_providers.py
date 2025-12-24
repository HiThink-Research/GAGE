"""Configuration schema for HuggingFace Inference Providers backend."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class InferenceProvidersBackendConfig(BackendConfigBase):
    """Config for HuggingFace Inference Providers (Together, Anyscale, etc.)."""

    provider: str = Field(..., description="Inference provider 名称，例如 together/anyscale/runpod 等")
    model_name: str = Field(..., description="提供给 provider 的模型 ID")
    tokenizer_name: Optional[str] = Field(
        default=None,
        description="可选 tokenizer 名称；若缺省则与 model_name 同步",
    )
    token: Optional[str] = Field(default=None, description="显式 HF API token（否则读取环境变量）")
    timeout: Optional[int] = Field(default=None, ge=1, description="请求超时秒数")
    proxies: Any | None = Field(default=None, description="代理配置，兼容 requests/AsyncInferenceClient")
    org_to_bill: Optional[str] = Field(default=None, description="计费组织标识")
    parallel_calls_count: int = Field(default=10, ge=1, description="并发调用上限（semaphore 控制）")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters,
        description="默认采样参数，映射到 provider 的 OpenAI 兼容字段",
    )
    http_retry_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="HTTP 重试配置，支持 attempts/interval/base_sleep/multiplier/max_retries",
    )
