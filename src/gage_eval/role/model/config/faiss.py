"""Faiss retrieval backend schema."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase


class FaissBackendConfig(BackendConfigBase):
    index_path: str
    vector_dim: int = Field(default=1536, ge=1)
    use_gpu: bool = False
    probe: Optional[int] = None
