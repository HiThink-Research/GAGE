"""Strict E2B provider configuration for AgentKit v2 environments."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator

from gage_eval.environment.contracts import DEFAULT_EXEC_STREAM_LIMIT_BYTES
from gage_eval.environment.resources import NetworkPolicy


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class E2BEnvironmentConfig(_StrictModel):
    template_id: str
    sandbox_timeout_s: float | None = Field(default=None, ge=0)
    request_timeout_s: float | None = Field(default=None, ge=0)
    network_policy: NetworkPolicy | None = None
    startup_env: dict[str, str] = Field(default_factory=dict)
    user: str | None = None
    stdout_limit_bytes: int = Field(default=DEFAULT_EXEC_STREAM_LIMIT_BYTES, ge=0)
    stderr_limit_bytes: int = Field(default=DEFAULT_EXEC_STREAM_LIMIT_BYTES, ge=0)

    @field_validator("template_id")
    @classmethod
    def _template_id_is_non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("must be non-empty")
        return value

    @field_validator("user")
    @classmethod
    def _optional_user_is_non_empty(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("must be non-empty")
        return value


__all__ = ["E2BEnvironmentConfig"]
