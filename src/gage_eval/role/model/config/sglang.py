"""SGLang backend configuration schema."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class SGLangBackendConfig(BackendConfigBase):
    base_url: str = Field(description="SGLang service URL")
    api_key: Optional[str] = None
    stream: bool = False
    timeout: int = Field(default=60, ge=1)
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
