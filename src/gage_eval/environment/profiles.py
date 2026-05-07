"""Provider-neutral environment profile records."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources


class EnvironmentProfile(BaseModel):
    """Cold-path profile selected by the compiled runtime plan."""

    model_config = ConfigDict(extra="forbid", protected_namespaces=(), strict=True)

    profile_id: str
    provider: str
    config: dict[str, Any] = Field(default_factory=dict)
    startup_env: dict[str, str] = Field(default_factory=dict)
    resources: EnvironmentResources | None = None
    metadata: dict[str, str] = Field(default_factory=dict)
    capabilities: EnvironmentCapabilities | None = None
