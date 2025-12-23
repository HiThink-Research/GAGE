"""Configs for vendor-specific HTTP APIs (Anthropic/Gemini/OpenAI Batch)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class ClaudeBackendConfig(BackendConfigBase):
    model: str = Field(default="claude-3-opus-20240229", description="Anthropic Claude 模型名称")
    api_key: Optional[str] = Field(default=None, description="Anthropic API Key（若为空则读取 ANTHROPIC_API_KEY）")
    base_url: Optional[str] = Field(default=None, description="自定义 API Base URL，默认为官方地址")
    max_output_tokens: int = Field(default=1024, ge=1, description="默认 max_tokens")
    temperature: float = Field(default=0.0, ge=0.0, description="默认 temperature")
    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    default_system: Optional[str] = Field(default=None, description="默认 system prompt")


class GeminiBackendConfig(BackendConfigBase):
    model: str = Field(default="gemini-1.5-pro", description="Gemini 模型版本")
    api_key: Optional[str] = Field(default=None, description="Google API Key（未提供时读取 GOOGLE_API_KEY）")
    safety_settings: Dict[str, str] = Field(default_factory=dict, description="安全策略覆盖")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters, description="默认采样参数"
    )


class OpenAIBatchBackendConfig(BackendConfigBase):
    model: str = Field(default="gpt-4o-mini", description="OpenAI 模型名")
    api_key: Optional[str] = Field(default=None, description="OpenAI API Key（或使用 OPENAI_API_KEY）")
    base_url: Optional[str] = Field(default=None, description="自定义 Base URL，例如 Azure 端点")
    poll_interval: int = Field(default=5, ge=1, description="轮询 batch 状态的间隔秒数")
    timeout: int = Field(default=600, ge=30, description="等待 batch 完成的最长期限（秒）")
    metadata: Dict[str, str] = Field(default_factory=dict, description="附加 metadata，写入 batch 创建请求")
    keep_files: bool = Field(default=False, description="是否在执行后保留上传/输出文件")
    extra_client_args: Dict[str, Any] = Field(default_factory=dict, description="OpenAI 客户端额外参数")
    generation_parameters: GenerationParameters = Field(
        default_factory=GenerationParameters, description="默认采样参数"
    )
