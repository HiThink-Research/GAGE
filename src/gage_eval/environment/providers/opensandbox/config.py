"""Strict reserved OpenSandbox provider configuration."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OpenSandboxReservedConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)


__all__ = ["OpenSandboxReservedConfig"]
