"""LiteLLM backend configuration."""

from __future__ import annotations

from typing import Dict, Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class LiteLLMBackendConfig(BackendConfigBase):
    model: str = Field(default="gpt-4o-mini")
    api_key: Optional[str] = None
    api_base: Optional[str] = None
    extra_headers: Dict[str, str] = Field(default_factory=dict)
    streaming: bool = False
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
