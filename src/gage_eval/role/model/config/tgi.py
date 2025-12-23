"""Text Generation Inference backend configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class TGIBackendConfig(BackendConfigBase):
    base_url: str
    model_id: Optional[str] = None
    timeout: int = Field(default=60, ge=1)
    trust_remote_code: bool = True
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
