"""Schema for vLLM backend configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase
from gage_eval.role.model.config.generations import GenerationParameters


class VLLMBackendConfig(BackendConfigBase):
    model_path: str = Field(description="Filesystem path or HuggingFace repo id")
    tensor_parallel_size: int = Field(default=1, ge=1)
    max_model_len: Optional[int] = Field(default=None, ge=128)
    dtype: Optional[str] = None
    trust_remote_code: bool = True
    output_type: str = Field(default="text", description="Controls output adapter behavior")
    generation_parameters: GenerationParameters = Field(default_factory=GenerationParameters)
