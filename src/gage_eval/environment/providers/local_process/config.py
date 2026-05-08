"""Strict local-process provider configuration for AgentKit v2 environments."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from gage_eval.environment.contracts import DEFAULT_EXEC_STREAM_LIMIT_BYTES


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


class LocalProcessEnvironmentConfig(_StrictModel):
    """Configuration for host-local subprocess environments.

    ``workdir`` uses an exact provider root. ``base_cwd`` asks the provider to
    create a fresh child directory under that base for each environment.
    """

    workdir: str | None = None
    base_cwd: str | None = None
    startup_env: dict[str, str] = Field(default_factory=dict)
    stdout_limit_bytes: int = Field(default=DEFAULT_EXEC_STREAM_LIMIT_BYTES, ge=0)
    stderr_limit_bytes: int = Field(default=DEFAULT_EXEC_STREAM_LIMIT_BYTES, ge=0)

    @field_validator("workdir", "base_cwd")
    @classmethod
    def _path_is_absolute(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip():
            raise ValueError("must be non-empty")
        if not Path(value).is_absolute():
            raise ValueError("must be absolute")
        return value

    @model_validator(mode="after")
    def _validate_workdir_selection(self) -> "LocalProcessEnvironmentConfig":
        if self.workdir is not None and self.base_cwd is not None:
            raise ValueError("workdir and base_cwd are mutually exclusive")
        return self


__all__ = ["LocalProcessEnvironmentConfig"]
