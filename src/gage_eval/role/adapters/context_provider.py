"""Context provider adapter (RAG/knowledge retriever)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.mcp.utils import sync_mcp_endpoint
from gage_eval.registry import ensure_async, registry
from gage_eval.sandbox.provider import SandboxProvider
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState


@registry.asset(
    "roles",
    "context_provider",
    desc="RAG/knowledge-augmented context provider role",
    tags=("role", "context"),
    role_type="context_provider",
)
class ContextProviderAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        *,
        implementation: str,
        implementation_params: Optional[Dict[str, Any]] = None,
        capabilities=(),
        role_type: str = "context_provider",
        mcp_client_id: Optional[str] = None,
        mcp_client: Optional[Any] = None,
        **_,
    ) -> None:
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=resolved_caps)
        if not implementation:
            raise ValueError("ContextProviderAdapter requires non-empty implementation")
        self._implementation = implementation
        self._implementation_params = dict(implementation_params or {})
        if mcp_client_id:
            self._implementation_params.setdefault("mcp_client_id", mcp_client_id)
        if mcp_client is not None:
            self._implementation_params.setdefault("mcp_client", mcp_client)
        try:
            impl_cls = registry.get("context_impls", implementation)
        except KeyError:
            registry.auto_discover("context_impls", "gage_eval.role.context")
            impl_cls = registry.get("context_impls", implementation)
        self._impl = impl_cls(**self._implementation_params)
        provider = getattr(self._impl, "aprovide", None) or getattr(self._impl, "provide", None)
        if provider is None:
            raise TypeError(f"context_impls '{implementation}' must define provide/aprovide")
        self._provider = ensure_async(provider)

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        impl_payload = dict(payload or {})
        sandbox_provider = impl_payload.get("sandbox_provider")
        mcp_client = self._implementation_params.get("mcp_client")
        if isinstance(sandbox_provider, SandboxProvider) and mcp_client is not None:
            runtime_handle = sandbox_provider.runtime_handle()
            sync_mcp_endpoint(mcp_client, runtime_handle)
        impl_payload["params"] = self._merge_params(payload)
        impl_payload.setdefault("implementation", self._implementation)
        impl_payload.setdefault("adapter_id", self.adapter_id)
        result = await self._provider(impl_payload, state)
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        return {"context": result}

    def _merge_params(self, payload: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        merged = dict(self._implementation_params)
        if not payload:
            return merged
        step = payload.get("step")
        step_params: Any = None
        if isinstance(step, dict):
            step_params = step.get("params") or step.get("args")
        elif step is not None:
            getter = getattr(step, "get", None)
            if callable(getter):
                try:
                    step_params = getter("params") or getter("args")
                except Exception:
                    step_params = None
            if step_params is None and hasattr(step, "params"):
                step_params = getattr(step, "params")
            if step_params is None and hasattr(step, "args"):
                step_params = getattr(step, "args")
        if isinstance(step_params, dict):
            merged.update(step_params)
        extra_params = payload.get("params")
        if isinstance(extra_params, dict):
            merged.update(extra_params)
        return merged
