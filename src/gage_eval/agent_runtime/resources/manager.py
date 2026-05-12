from __future__ import annotations

import asyncio
import inspect
import threading
from dataclasses import dataclass
from typing import Any, Literal, Mapping

from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.environment.lease import EnvironmentLease
from gage_eval.environment.manager import EnvironmentManager
from gage_eval.environment.profiles import EnvironmentProfile
from gage_eval.environment.providers.registry import create_default_provider_registry
from gage_eval.environment.resources import EnvironmentCapabilities, EnvironmentResources
from gage_eval.observability.trace import ObservabilityTrace


@dataclass
class RuntimeLeaseBinding:
    """Carries the acquired runtime resource and environment lease."""

    resource_lease: ResourceLease | None
    environment_lease: EnvironmentLease | None = None
    environment_manager: EnvironmentManager | None = None
    owns_environment_lease: bool = True


class RuntimeResourceManager:
    """Acquires local-first resource leases for agent runtimes."""

    def __init__(
        self,
        legacy_manager: Any | None = None,
        *,
        environment_manager: EnvironmentManager | None = None,
    ) -> None:
        del legacy_manager
        self._environment_manager = environment_manager

    def acquire(
        self,
        session: AgentRuntimeSession,
        *,
        resource_plan: dict[str, Any],
        trace: ObservabilityTrace | None = None,
        sample: dict[str, Any] | None = None,
    ) -> RuntimeLeaseBinding:
        """Acquire the resource lease declared by the compiled plan."""

        if "sandbox_config" in resource_plan:
            raise ValueError("config.legacy_key.sandbox_config: use environment_profile/provider_config")
        if isinstance(sample, Mapping) and "sandbox" in sample:
            raise ValueError("sample.legacy_key.sandbox: use metadata.environment_overrides")
        profile_payload = _environment_profile_payload(resource_plan)
        provider_config = _provider_config_payload(resource_plan, profile_payload)
        if not profile_payload and not provider_config and not resource_plan.get("resource_kind"):
            return RuntimeLeaseBinding(
                resource_lease=None,
            )

        resource_kind = _resource_kind_for_plan(resource_plan, profile_payload)
        return self._acquire_environment_provider(
            session,
            resource_plan=resource_plan,
            profile_payload=profile_payload,
            provider_config=provider_config,
            resource_kind=resource_kind,
            sample=sample,
        )

    def _acquire_environment_provider(
        self,
        session: AgentRuntimeSession,
        *,
        resource_plan: dict[str, Any],
        profile_payload: dict[str, Any],
        provider_config: dict[str, Any],
        resource_kind: str,
        sample: dict[str, Any] | None = None,
    ) -> RuntimeLeaseBinding:
        environment_manager = self._get_environment_manager()
        profile_id = _profile_id_from_profile(profile_payload, session=session)
        provider = _environment_provider_for_resource(resource_kind, profile_payload)
        provider_config = _resolve_provider_config(
            resource_plan=resource_plan,
            sample=sample,
            base_provider_config=provider_config,
            provider=provider,
            profile_id=profile_id,
        )
        resources = _environment_resources_from_plan(resource_plan, profile_payload)
        startup_env = _startup_env_from_plan(resource_plan, profile_payload)
        lifecycle = _lifecycle_from_plan(resource_plan, profile_payload)
        profile = EnvironmentProfile(
            profile_id=profile_id,
            provider=provider,
            config=dict(provider_config),
            startup_env=dict(startup_env),
            resources=resources,
            capabilities=_environment_capabilities_from_profile(profile_payload),
            metadata=_environment_profile_metadata(profile_payload, session=session),
        )
        lease_metadata = {
            "run_id": session.run_id,
            "task_id": session.task_id,
            "sample_id": session.sample_id,
            "benchmark_kit_id": session.benchmark_kit_id,
            "scheduler_type": session.scheduler_type,
        }
        lease_metadata.update(_workdir_metadata(provider_config))
        environment_lease = _run_async_blocking(
            environment_manager.acquire(
                kit_id=session.benchmark_kit_id,
                provider=provider,
                profile_id=profile_id,
                profile=profile,
                provider_config=dict(provider_config),
                resources=resources,
                startup_env=startup_env,
                lifecycle=lifecycle,
                metadata=lease_metadata,
            )
        )
        resource_lease = _build_environment_resource_lease(
            session=session,
            resource_plan=resource_plan,
            environment_profile=profile.model_dump(mode="python", exclude_none=True),
            provider_config=provider_config,
            environment_lease=environment_lease,
            resource_kind=resource_kind,
        )
        return RuntimeLeaseBinding(
            resource_lease=resource_lease,
            environment_lease=environment_lease,
            environment_manager=environment_manager,
            owns_environment_lease=True,
        )

    def _get_environment_manager(self) -> EnvironmentManager:
        if self._environment_manager is None:
            self._environment_manager = EnvironmentManager(
                registry=create_default_provider_registry(),
            )
        return self._environment_manager

    def bind_existing(
        self,
        session: AgentRuntimeSession,
        *,
        resource_plan: dict[str, Any],
        environment_lease: EnvironmentLease,
    ) -> RuntimeLeaseBinding:
        """Wrap an externally managed environment lease."""

        if "sandbox_config" in resource_plan:
            raise ValueError("config.legacy_key.sandbox_config: use environment_profile/provider_config")
        profile_payload = _environment_profile_payload(resource_plan)
        provider_config = _provider_config_payload(resource_plan, profile_payload)
        resource_kind = _resource_kind_for_plan(resource_plan, profile_payload)
        lease = _build_environment_resource_lease(
            session=session,
            resource_plan=resource_plan,
            environment_profile=profile_payload,
            provider_config=provider_config,
            environment_lease=environment_lease,
            resource_kind=resource_kind,
        )
        return RuntimeLeaseBinding(
            resource_lease=lease,
            environment_lease=environment_lease,
            environment_manager=None,
            owns_environment_lease=False,
        )

    @staticmethod
    def release(binding: RuntimeLeaseBinding) -> None:
        """Release the resource lease if it exists."""

        if (
            binding.environment_lease is not None
            and binding.environment_manager is not None
            and binding.owns_environment_lease
        ):
            _run_async_blocking(binding.environment_manager.release(binding.environment_lease))


def _resource_kind_for_plan(resource_plan: dict[str, Any], profile_payload: dict[str, Any]) -> str:
    return str(
        resource_plan.get("resource_kind")
        or resource_plan.get("environment_provider")
        or profile_payload.get("provider")
        or "docker"
    )


def _environment_profile_payload(resource_plan: dict[str, Any]) -> dict[str, Any]:
    profile = resource_plan.get("environment_profile")
    if hasattr(profile, "model_dump"):
        dumped = profile.model_dump(mode="python", exclude_none=True)
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return dict(profile) if isinstance(profile, Mapping) else {}


def _provider_config_payload(
    resource_plan: dict[str, Any],
    profile_payload: dict[str, Any],
) -> dict[str, Any]:
    provider_config = resource_plan.get("provider_config")
    if isinstance(provider_config, Mapping):
        return dict(provider_config)
    profile_provider_config = profile_payload.get("provider_config")
    if isinstance(profile_provider_config, Mapping):
        return dict(profile_provider_config)
    profile_config = profile_payload.get("config")
    if isinstance(profile_config, Mapping):
        return dict(profile_config)
    return {}


def _build_environment_resource_lease(
    *,
    session: AgentRuntimeSession,
    resource_plan: dict[str, Any],
    environment_profile: dict[str, Any],
    provider_config: dict[str, Any],
    environment_lease: EnvironmentLease,
    resource_kind: str,
) -> ResourceLease:
    descriptor = environment_lease.to_descriptor()
    return ResourceLease(
        lease_id=environment_lease.lease_id,
        resource_kind=resource_kind,  # type: ignore[arg-type]
        profile_id=environment_lease.profile_id,
        lifecycle=environment_lease.lifecycle,
        endpoints={},
        handle_ref=descriptor,
        cleanup_policy=dict(resource_plan.get("cleanup_policy") or {}),
        metadata={
            "environment_profile": dict(environment_profile),
            "provider_config": dict(provider_config),
            "environment_descriptor": descriptor,
            "run_id": session.run_id,
            "task_id": session.task_id,
            "sample_id": session.sample_id,
        },
    )


def _profile_id_from_profile(
    profile_payload: dict[str, Any],
    *,
    session: AgentRuntimeSession,
) -> str:
    return str(
        profile_payload.get("profile_id")
        or profile_payload.get("template_name")
        or profile_payload.get("provider")
        or session.benchmark_kit_id
    )


def _lifecycle_from_plan(
    resource_plan: dict[str, Any],
    profile_payload: dict[str, Any],
) -> Literal["per_sample", "per_task"]:
    lifecycle = str(resource_plan.get("lifecycle") or profile_payload.get("lifecycle") or "per_sample")
    if lifecycle == "per_task":
        return "per_task"
    return "per_sample"


def _startup_env_from_plan(
    resource_plan: dict[str, Any],
    profile_payload: dict[str, Any],
) -> dict[str, str]:
    startup_env = resource_plan.get("startup_env") or profile_payload.get("startup_env")
    if not isinstance(startup_env, dict):
        return {}
    return {str(key): str(value) for key, value in startup_env.items()}


def _environment_resources_from_plan(
    resource_plan: dict[str, Any],
    profile_payload: dict[str, Any],
) -> EnvironmentResources:
    resources = resource_plan.get("resources") or profile_payload.get("resources")
    if hasattr(resources, "model_dump"):
        return EnvironmentResources.model_validate(resources.model_dump(mode="python", exclude_none=True))
    if not isinstance(resources, dict):
        return EnvironmentResources()
    return EnvironmentResources.model_validate(resources)


def _environment_capabilities_from_profile(
    profile_payload: dict[str, Any],
) -> EnvironmentCapabilities | None:
    capabilities = profile_payload.get("capabilities")
    if hasattr(capabilities, "model_dump"):
        return EnvironmentCapabilities.model_validate(capabilities.model_dump(mode="python", exclude_none=True))
    if not isinstance(capabilities, dict):
        return None
    return EnvironmentCapabilities.model_validate(capabilities)


def _environment_profile_metadata(
    profile_payload: dict[str, Any],
    *,
    session: AgentRuntimeSession,
) -> dict[str, str]:
    metadata = {"kit_id": session.benchmark_kit_id}
    asset_dir = profile_payload.get("asset_dir")
    if isinstance(asset_dir, str) and asset_dir:
        metadata["asset_dir"] = asset_dir
    raw_metadata = profile_payload.get("metadata")
    if isinstance(raw_metadata, Mapping):
        metadata.update({str(key): str(value) for key, value in raw_metadata.items()})
    return metadata


def _resolve_provider_config(
    *,
    resource_plan: dict[str, Any],
    sample: dict[str, Any] | None,
    base_provider_config: dict[str, Any],
    provider: str,
    profile_id: str,
) -> dict[str, Any]:
    resolver = resource_plan.get("provider_config_resolver")
    if not callable(resolver):
        return dict(base_provider_config)
    kwargs = {
        "sample": sample or {},
        "base_provider_config": dict(base_provider_config),
        "provider": provider,
        "profile_id": profile_id,
    }
    try:
        signature = inspect.signature(resolver)
    except (TypeError, ValueError):
        selected_kwargs = kwargs
    else:
        if any(parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
            selected_kwargs = kwargs
        else:
            selected_kwargs = {
                key: value
                for key, value in kwargs.items()
                if key in signature.parameters
            }
    return dict(
        resolver(**selected_kwargs)
    )


def _workdir_metadata(provider_config: Mapping[str, Any]) -> dict[str, str]:
    metadata: dict[str, str] = {}
    for key in ("exec_workdir", "workdir"):
        value = provider_config.get(key)
        if isinstance(value, str) and value.startswith("/"):
            metadata[key] = value
    return metadata


def _environment_provider_for_resource(resource_kind: str, profile_payload: dict[str, Any]) -> str:
    provider = profile_payload.get("provider")
    if isinstance(provider, str) and provider.strip():
        return provider.strip()
    return resource_kind


def _run_async_blocking(awaitable: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)

    result: dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = asyncio.run(awaitable)
        except BaseException as exc:  # pragma: no cover - reraised in caller thread
            result["error"] = exc

    thread = threading.Thread(target=_runner, daemon=True)
    thread.start()
    thread.join()
    if "error" in result:
        raise result["error"]
    return result.get("value")
