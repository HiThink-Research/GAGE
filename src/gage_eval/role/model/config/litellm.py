"""LiteLLM backend configuration."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class LiteLLMBackendConfig(BackendConfigBase):
    model: str = Field(default="gpt-4o-mini")
    provider: Optional[str] = None
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    streaming: bool = False
    timeout: Optional[float] = Field(default=None, ge=0.0)
    max_retries: int = Field(default=6, ge=1)
    retry_sleep: float = Field(default=1.0, ge=0.0)
    retry_multiplier: float = Field(default=2.0, ge=1.0)
    max_model_length: Optional[int] = Field(default=None, ge=1)
    prefer_litellm_kimi: bool = False
    force_kimi_direct: bool = False
    verbose: bool = False
    mock_api_base: Optional[str] = Field(
        default=None, description="用于本地/自建 OpenAI 兼容服务的 base_url（无额度时模拟调用）"
    )
    mock_api_key: Optional[str] = Field(default=None, description="本地/自建服务的密钥（可留空走环境变量）")
    mock_model: Optional[str] = Field(default=None, description="模拟调用时使用的模型名；缺省沿用 model")
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
