"""DUT agent adapter thin entrypoint for AgentRuntimeExecutor."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.sample import Sample, sample_to_dict
from gage_eval.registry import registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


class DUTAgentConfigurationError(RuntimeError):
    """Raised when the DUT agent adapter is not wired to runtime execution."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


@registry.asset(
    "roles",
    "dut_agent",
    desc="DUT agent adapter with tool-calling capabilities",
    tags=("role", "agent"),
    role_type="dut_agent",
)
class DUTAgentAdapter(RoleAdapter):
    """Adapter facade that delegates DUT execution to AgentRuntimeExecutor."""

    def __init__(
        self,
        adapter_id: str,
        role_type: str,
        capabilities,
        *,
        agent_backend: Any | None = None,
        prompt_renderer: Optional[Any] = None,
        sandbox_manager: Optional[Any] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_router: Optional[Any] = None,
        mcp_clients: Optional[Dict[str, Any]] = None,
        human_gateway: Optional[Any] = None,
        agent_runtime_id: Optional[str] = None,
        compat_runtime_id: Optional[str] = None,
        executor_ref: Optional[Any] = None,
        max_turns: int = 8,
        **params,
    ) -> None:
        del (
            agent_backend,
            prompt_renderer,
            sandbox_manager,
            sandbox_profiles,
            tool_router,
            mcp_clients,
            human_gateway,
        )
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=capabilities,
            resource_requirement=params.pop("resource_requirement", None),
            sandbox_config=params.pop("sandbox_config", None),
        )
        self.agent_runtime_id = agent_runtime_id
        self.compat_runtime_id = compat_runtime_id
        self.executor_ref = executor_ref
        self.params = dict(params)
        self.max_turns = max(1, int(max_turns))

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        del state
        if self.executor_ref is None:
            raise DUTAgentConfigurationError(
                "dut_agent.executor_ref.missing",
                "DUTAgentAdapter requires executor_ref and no longer supports legacy agent_backend fallback.",
            )
        raw_sample = payload.get("sample")
        sample = _runtime_sample_dict(raw_sample)
        runtime_payload = payload
        if raw_sample is not None and sample is not raw_sample:
            runtime_payload = dict(payload)
            runtime_payload["sample"] = sample
        return await self.executor_ref.aexecute(
            sample=sample,
            payload=runtime_payload,
            trace=payload.get("trace"),
        )

    def shutdown(self) -> None:
        """Release executor-owned resources."""

        issues: list[Exception] = []
        try:
            resource_manager = getattr(self.executor_ref, "resource_manager", None)
            sandbox_manager = getattr(resource_manager, "_sandbox_manager", None)
            if sandbox_manager is None:
                sandbox_manager = getattr(resource_manager, "sandbox_manager", None)
            shutdown_fn = getattr(sandbox_manager, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            issues.append(exc)

        if issues:
            raise RuntimeError(
                "; ".join(f"{type(issue).__name__}: {issue}" for issue in issues)
            )


def _runtime_sample_dict(sample: Any) -> dict[str, Any]:
    if isinstance(sample, dict):
        return sample
    if isinstance(sample, Sample):
        return sample_to_dict(sample)
    if sample is None:
        return {}
    return dict(sample) if hasattr(sample, "items") else {}
