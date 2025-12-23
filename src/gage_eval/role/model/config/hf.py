"""Schema for HuggingFace Transformers backend."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class HFBackendConfig(BackendConfigBase):
    model_name_or_path: str
    trust_remote_code: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None
    max_batch_size: Optional[int] = Field(default=None, ge=1)
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
