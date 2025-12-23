"""Generic HTTP backend configuration."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class HTTPBackendConfig(BackendConfigBase):
    base_url: str = Field(description="HTTP endpoint that implements a text generation API")
    timeout: int = Field(default=60, ge=1)
    headers: Dict[str, str] = Field(default_factory=dict)
    model: Optional[str] = None
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)


class MultiProviderHTTPBackendConfig(BackendConfigBase):
    provider: str = Field(description="Provider name registered on HuggingFace Inference Providers")
    model_name: str = Field(description="Model identifier exposed by the provider")
    tokenizer_name: Optional[str] = Field(default=None, description="Tokenizer used for chat template rendering")
    timeout: Optional[int] = Field(default=None, ge=1)
    proxies: Dict[str, str] = Field(default_factory=dict)
    org_to_bill: Optional[str] = None
    token: Optional[str] = Field(default=None, description="HF API token (fallback to HF_API_TOKEN env)")
    parallel_calls_count: int = Field(default=4, ge=1)
    http_retry_params: Dict[str, Any] = Field(default_factory=dict)
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
