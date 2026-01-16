"""Judge extension adapter (delegates to judge_impls)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.registry import ensure_async, registry
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "roles",
    "judge_extend",
    desc="Judge extension role adapter",
    tags=("role", "judge"),
    role_type="judge_extend",
)
class JudgeExtendAdapter(RoleAdapter):
    def __init__(
        self,
        adapter_id: str,
        *,
        implementation: str,
        implementation_params: Optional[Dict[str, Any]] = None,
        capabilities=(),
        role_type: str = "judge_extend",
        **_,
    ) -> None:
        resolved_caps = tuple(capabilities) if capabilities else ("text",)
        super().__init__(adapter_id=adapter_id, role_type=role_type, capabilities=resolved_caps)
        if not implementation:
            raise ValueError("JudgeExtendAdapter requires non-empty implementation")
        self._implementation = implementation
        self._implementation_params = dict(implementation_params or {})
        impl_cls = registry.get("judge_impls", implementation)
        self._impl = impl_cls(**self._implementation_params)
        provider = getattr(self._impl, "ainvoke", None) or getattr(self._impl, "invoke", None)
        if provider is None:
            raise TypeError(f"judge_impls '{implementation}' must define invoke/ainvoke")
        self._provider = ensure_async(provider)

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        impl_payload = dict(payload or {})
        sandbox_provider = impl_payload.get("sandbox_provider")
        if isinstance(sandbox_provider, SandboxProvider) and "runtime_handle" not in impl_payload:
            handle = sandbox_provider.get_handle()
            impl_payload["runtime_handle"] = handle.runtime_handle if handle else {}
        impl_payload["params"] = self._merge_params(payload)
        impl_payload.setdefault("implementation", self._implementation)
        impl_payload.setdefault("adapter_id", self.adapter_id)
        result = await self._provider(impl_payload, state)
        if result is None:
            return {}
        if isinstance(result, dict):
            return result
        return {"result": result}

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
