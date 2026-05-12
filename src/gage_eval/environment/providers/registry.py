"""Environment provider registry for AgentKit v2."""

from __future__ import annotations

from typing import Any, Protocol, TypeVar, runtime_checkable

from gage_eval.environment.contracts import BaseEnvironment
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.resources import EnvironmentResources


@runtime_checkable
class EnvironmentProvider(Protocol):
    """Minimal provider surface consumed by EnvironmentManager."""

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
        ...

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
        ...


ProviderT = TypeVar("ProviderT")


class ProviderRegistry:
    """Explicit in-process registry with no provider SDK imports."""

    def __init__(self) -> None:
        self._providers: dict[str, Any] = {}

    def register(self, provider_id: str, provider: ProviderT) -> ProviderT:
        if not provider_id:
            raise ValueError("provider_id must be non-empty")
        self._providers[provider_id] = provider
        return provider

    def get(self, provider_id: str) -> Any:
        try:
            return self._providers[provider_id]
        except KeyError as exc:
            raise KeyError(f"environment.provider.unregistered provider={provider_id}") from exc

    def registered_provider_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._providers))


def create_default_provider_registry() -> ProviderRegistry:
    """Create a registry with built-in provider factories.

    The constructor stays intentionally empty so tests and callers can keep
    explicit registration semantics. This factory is the opt-in default set.
    """

    from gage_eval.environment.providers.docker import DockerEnvironmentProvider
    from gage_eval.environment.providers.e2b import E2BEnvironmentProvider
    from gage_eval.environment.providers.local_process import LocalProcessEnvironmentProvider
    from gage_eval.environment.providers.opensandbox import OpenSandboxReservedProvider

    registry = ProviderRegistry()
    registry.register("docker", DockerEnvironmentProvider())
    registry.register("e2b", E2BEnvironmentProvider())
    registry.register("local_process", LocalProcessEnvironmentProvider())
    registry.register("opensandbox", OpenSandboxReservedProvider())
    return registry
