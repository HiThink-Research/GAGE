"""OpenAI compatible backend schema."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class OpenAIHTTPBackendConfig(BackendConfigBase):
    base_url: str = Field(default="https://api.openai.com/v1")
    api_key: Optional[str] = Field(default=None, description="Defaults to OPENAI_API_KEY environment variable")
    model: str = Field(default="gpt-4o-mini")
    organization: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    timeout: int = Field(default=120, ge=1)
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
