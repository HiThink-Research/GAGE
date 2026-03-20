"""Runtime strategy selection for role invocation."""

from __future__ import annotations

from typing import Any, Dict, Tuple


class AdapterInvokeRuntime:
    """Lightweight runtime wrapper around adapter invocation."""

    runtime_mode = "native"
    strategy_id = "native_runtime"

    def __init__(self, adapter: Any) -> None:
        self._adapter = adapter

    def execute(self, payload: Dict[str, Any], trace) -> Dict[str, Any]:
        """Execute the adapter with a fresh sample-local state."""

        del trace
        prepared_payload = dict(payload or {})
        route = dict(prepared_payload.get("runtime_route") or {})
        route.setdefault("runtime_mode", self.runtime_mode)
        route.setdefault("strategy_id", self.strategy_id)
        prepared_payload["runtime_route"] = route
        state = self._adapter.clone_for_sample()
        return self._adapter.invoke(prepared_payload, state)


class NativeRoleRuntime(AdapterInvokeRuntime):
    """Direct in-process runtime for native backends and non-model adapters."""

    runtime_mode = "native"
    strategy_id = "native_runtime"


class HttpRoleRuntime(AdapterInvokeRuntime):
    """Managed runtime wrapper for HTTP-style backends."""

    runtime_mode = "http"
    strategy_id = "http_runtime"

    def execute(self, payload: Dict[str, Any], trace) -> Dict[str, Any]:
        prepared_payload = dict(payload or {})
        prepared_payload.setdefault("runtime_transport", "http")
        return super().execute(prepared_payload, trace)


class RemoteRoleRuntime(AdapterInvokeRuntime):
    """Managed runtime wrapper for remote backends."""

    runtime_mode = "remote"
    strategy_id = "remote_runtime"

    def execute(self, payload: Dict[str, Any], trace) -> Dict[str, Any]:
        prepared_payload = dict(payload or {})
        prepared_payload.setdefault("runtime_transport", "remote")
        return super().execute(prepared_payload, trace)


class RuntimeStrategyFactory:
    """Build runtime wrappers from adapter/backend execution mode."""

    def build(self, adapter: Any) -> Tuple[AdapterInvokeRuntime, str, str]:
        """Return runtime object, runtime_mode, and strategy_id."""

        backend = getattr(adapter, "backend", None)
        runtime_mode = str(getattr(backend, "execution_mode", "native") or "native")
        runtime_mode = runtime_mode.strip().lower() or "native"
        if runtime_mode == "http":
            runtime = HttpRoleRuntime(adapter)
        elif runtime_mode == "remote":
            runtime = RemoteRoleRuntime(adapter)
        else:
            runtime_mode = "native"
            runtime = NativeRoleRuntime(adapter)
        return runtime, runtime.runtime_mode, runtime.strategy_id
