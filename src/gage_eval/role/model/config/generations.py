"""Shared generation parameter schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GenerationParameters(BaseModel):
    """Common generation parameter set shared across backends."""

    max_new_tokens: Optional[int] = Field(default=512, ge=1)
    temperature: Optional[float] = Field(default=0.7, ge=0.0)
    top_p: Optional[float] = Field(default=0.95, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(default=None, ge=1)
    repetition_penalty: Optional[float] = Field(default=None, ge=0.0)
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    stop: Optional[List[str]] = None
    stop_token_ids: Optional[List[int]] = None
    logprobs: Optional[bool] = None
    best_of: Optional[int] = Field(default=None, ge=1)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    top_logprobs: Optional[int] = Field(default=None, ge=1)

    def stop_sequences(self) -> Optional[List[str]]:
        return self.stop

    def to_transformers_dict(self) -> Dict[str, Any]:
        """Translate stored sampling parameters to transformers-friendly kwargs."""

        params = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "min_p": self.min_p,
        }
        return {k: v for k, v in params.items() if v is not None}

    def to_inference_providers_dict(self) -> Dict[str, Any]:
        """Translate parameters for HF Inference Providers (OpenAI-compatible)."""
        params = {
            "max_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "presence_penalty": self.presence_penalty,
            "frequency_penalty": self.frequency_penalty,
            "stop": self.stop,
        }
        return {k: v for k, v in params.items() if v is not None}

    def to_dict(self) -> Dict[str, Any]:
        """Return a shallow copy excluding `None` entries."""

        return {k: v for k, v in self.model_dump().items() if v is not None}
