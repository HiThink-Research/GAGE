"""FlagEmbedding backend config."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase


class FlagEmbeddingBackendConfig(BackendConfigBase):
    model_name: str = Field(default="BAAI/bge-large-en")
    batch_size: int = Field(default=16, ge=1)
    device: Optional[str] = None
