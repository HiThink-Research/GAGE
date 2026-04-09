from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from gage_eval.agent_runtime.resources.contracts import ResourceLease
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.sandbox.manager import SandboxHandle, SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


@dataclass
class RuntimeLeaseBinding:
    """Carries the acquired lease plus the lazy sandbox provider."""

    resource_lease: ResourceLease | None
    sandbox_provider: SandboxProvider | None
    sandbox_handle: SandboxHandle | None


class RuntimeResourceManager:
    """Acquires local-first resource leases for agent runtimes."""

    def __init__(self, sandbox_manager: SandboxManager | None = None) -> None:
        self._sandbox_manager = sandbox_manager or SandboxManager()

    def acquire(
        self,
        session: AgentRuntimeSession,
        *,
        resource_plan: dict[str, Any],
        trace: ObservabilityTrace | None = None,
    ) -> RuntimeLeaseBinding:
        """Acquire the resource lease declared by the compiled plan."""

        # STEP 1: Resolve the sample-scoped provider from the resource plan.
        sandbox_config = dict(resource_plan.get("sandbox_config") or {})
        if not sandbox_config:
            return RuntimeLeaseBinding(
                resource_lease=None,
                sandbox_provider=None,
                sandbox_handle=None,
            )
        sandbox_config = self._sandbox_manager.resolve_config(sandbox_config)

        scope = SandboxScope(
            run_id=session.run_id,
            task_id=session.task_id,
            sample_id=session.sample_id,
        )
        provider = SandboxProvider(
            self._sandbox_manager,
            sandbox_config,
            scope,
            trace=trace,
        )

        # STEP 2: Materialize the handle immediately so failures stay in this stage.
        handle = provider.get_handle()
        runtime_handle = handle.runtime_handle if handle is not None else {}
        resource_kind = str(resource_plan.get("resource_kind") or _resolve_resource_kind(sandbox_config))
        lease = ResourceLease(
            lease_id=f"lease-{uuid4().hex}",
            resource_kind=resource_kind,  # type: ignore[arg-type]
            profile_id=str(
                sandbox_config.get("sandbox_id")
                or sandbox_config.get("template_name")
                or sandbox_config.get("runtime")
                or session.benchmark_kit_id
            ),
            lifecycle=str(sandbox_config.get("lifecycle") or "per_sample"),  # type: ignore[arg-type]
            endpoints=_extract_endpoints(runtime_handle),
            handle_ref=dict(runtime_handle or {}),
            cleanup_policy=dict(resource_plan.get("cleanup_policy") or {}),
            metadata={"sandbox_config": sandbox_config},
        )
        return RuntimeLeaseBinding(
            resource_lease=lease,
            sandbox_provider=provider,
            sandbox_handle=handle,
        )

    @staticmethod
    def release(binding: RuntimeLeaseBinding) -> None:
        """Release the resource lease if it exists."""

        if binding.sandbox_provider is not None:
            binding.sandbox_provider.release()


def _resolve_resource_kind(sandbox_config: dict[str, Any]) -> str:
    runtime = str(sandbox_config.get("runtime") or sandbox_config.get("backend") or "docker")
    if runtime == "tau2":
        return "local_process"
    return "docker"


def _extract_endpoints(runtime_handle: dict[str, Any]) -> dict[str, str]:
    endpoints: dict[str, str] = {}
    for key in ("env_endpoint", "environment_endpoint", "apis_endpoint", "mcp_endpoint"):
        value = runtime_handle.get(key)
        if value:
            endpoints[key] = str(value)
    return endpoints
