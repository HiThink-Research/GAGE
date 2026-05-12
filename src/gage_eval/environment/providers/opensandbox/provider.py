"""Reserved OpenSandbox provider stub for AgentKit v2."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ValidationError

from gage_eval.environment.contracts import BaseEnvironment
from gage_eval.environment.errors import EnvironmentCreateError, EnvironmentPreflightError
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.resources import EnvironmentResources

from .config import OpenSandboxReservedConfig


UNSUPPORTED_MESSAGE = (
    "opensandbox.provider.unsupported reserved provider stub; implementation is not available in this phase"
)


class OpenSandboxReservedProvider:
    """Discoverable placeholder that fails explicitly when used."""

    retry_budget_by_failure = {EnvironmentCreateError: 0}

    async def preflight(
        self,
        *,
        kit_id: str,
        provider: str,
        profile_id: str,
        profile: EnvironmentProfile,
        provider_config: Any,
        resources: EnvironmentResources,
        startup_env: dict[str, str],
        lifecycle: str,
        metadata: dict[str, Any],
    ) -> None:
        del kit_id, provider, profile_id, profile, resources, startup_env, lifecycle, metadata
        _coerce_config(provider_config)

    async def create(
        self,
        *,
        kit_id: str,
        provider: str,
        profile_id: str,
        profile: EnvironmentProfile,
        provider_config: Any,
        resources: EnvironmentResources,
        startup_env: dict[str, str],
        lifecycle: str,
        metadata: dict[str, Any],
    ) -> BaseEnvironment:
        del kit_id, provider, profile_id, profile, provider_config, resources, startup_env, lifecycle, metadata
        raise EnvironmentCreateError(UNSUPPORTED_MESSAGE)

    async def acquire(self, **request: Any) -> BaseEnvironment:
        del request
        raise EnvironmentCreateError(UNSUPPORTED_MESSAGE)


def _coerce_config(provider_config: Any) -> OpenSandboxReservedConfig:
    if isinstance(provider_config, OpenSandboxReservedConfig):
        return provider_config
    if isinstance(provider_config, BaseModel):
        raw = provider_config.model_dump(mode="python", exclude_none=True)
    elif isinstance(provider_config, dict):
        raw = dict(provider_config)
    else:
        raw = {}
    try:
        return OpenSandboxReservedConfig.model_validate(raw)
    except ValidationError as exc:
        raise EnvironmentPreflightError(f"opensandbox.config validation failed: {exc}") from None


__all__ = ["OpenSandboxReservedProvider"]
