"""Whisper/faster-whisper backend configuration."""

from __future__ import annotations

from typing import Optional

from pydantic import Field

from gage_eval.role.model.config.base import BackendConfigBase


class WhisperASRBackendConfig(BackendConfigBase):
    model_size: str = Field(default="large-v3")
    device: Optional[str] = None
    compute_type: str = Field(default="float16")
    beam_size: int = Field(default=5, ge=1)
    vad_filter: bool = True
