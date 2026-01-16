"""SandboxProvider for sample-scoped sandbox lifecycle management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from gage_eval.observability.trace import ObservabilityTrace

from gage_eval.sandbox.manager import SandboxHandle, SandboxManager


@dataclass(frozen=True)
class SandboxScope:
    """Scope metadata used to derive sandbox pool keys."""

    run_id: Optional[str] = None
    task_id: Optional[str] = None
    sample_id: Optional[str] = None
    arena_id: Optional[str] = None


class SandboxProvider:
    """Provide a lazily initialized sandbox handle for a sample.

    Args:
        manager: SandboxManager used to acquire/release runtimes.
        sandbox_config: Resolved sandbox configuration for the sample.
        scope: Scope metadata used for lifecycle-based pooling.
    """

    def __init__(
        self,
        manager: SandboxManager,
        sandbox_config: Optional[Dict[str, Any]],
        scope: SandboxScope,
        trace: Optional[ObservabilityTrace] = None,
    ) -> None:
        self._manager = manager
        self._sandbox_config = dict(sandbox_config or {})
        self._scope = scope
        self._handle: Optional[SandboxHandle] = None
        self._trace = trace

    @property
    def sandbox_config(self) -> Dict[str, Any]:
        """Return a copy of the resolved sandbox configuration."""

        return dict(self._sandbox_config)

    def get_handle(self) -> Optional[SandboxHandle]:
        """Return the sandbox handle, lazily starting the sandbox when needed."""

        if not self._sandbox_config:
            return None
        if self._handle is None:
            config = _prepare_sandbox_config(self._sandbox_config, self._scope)
            pool_key = config.get("pool_key")
            self._emit_event(
                "sandbox_provider_cache_miss",
                _build_provider_payload(config, self._scope, pool_key),
            )
            self._handle = self._manager.acquire(config, trace=self._trace, sample_id=self._scope.sample_id)
        else:
            self._emit_event(
                "sandbox_provider_cache_hit",
                _build_provider_payload(self._handle.config, self._scope, self._handle.pool_key),
            )
        return self._handle

    def runtime_handle(self) -> Dict[str, Any]:
        """Return the runtime handle if available, otherwise an empty dict."""

        handle = self.get_handle()
        if handle and handle.runtime_handle:
            return dict(handle.runtime_handle)
        return {}

    def release(self) -> None:
        """Release the sandbox handle if it was acquired."""

        if self._handle:
            payload = _build_provider_payload(self._handle.config, self._scope, self._handle.pool_key)
            self._emit_event("sandbox_release_start", payload)
            start = _now_ms()
            try:
                self._manager.release(self._handle)
            except Exception:
                payload = dict(payload)
                payload.update({"status": "failed", "latency_ms": _now_ms() - start})
                self._emit_event("sandbox_release_end", payload)
                raise
            payload = dict(payload)
            payload.update({"status": "success", "latency_ms": _now_ms() - start})
            self._emit_event("sandbox_release_end", payload)
            self._handle = None

    def _emit_event(self, event: str, payload: Dict[str, Any]) -> None:
        if self._trace is None:
            return
        self._trace.emit(event, payload, sample_id=self._scope.sample_id)


def _build_pool_key(config: Dict[str, Any], scope: SandboxScope) -> Optional[str]:
    explicit = config.get("pool_key")
    if explicit:
        return str(explicit)
    lifecycle = str(config.get("lifecycle", "per_sample"))
    sandbox_id = _resolve_sandbox_id(config)
    if lifecycle == "per_sample":
        return None
    if lifecycle == "per_profile":
        return sandbox_id
    scope_value = _resolve_scope_value(lifecycle, scope)
    if not sandbox_id:
        return scope_value
    if scope_value:
        return f"{sandbox_id}:{scope_value}"
    return sandbox_id


def _resolve_sandbox_id(config: Dict[str, Any]) -> Optional[str]:
    for key in ("sandbox_id", "template_name", "runtime"):
        value = config.get(key)
        if value:
            return str(value)
    return None


def _resolve_scope_value(lifecycle: str, scope: SandboxScope) -> Optional[str]:
    if lifecycle == "per_task":
        return scope.task_id or scope.sample_id
    if lifecycle == "per_run":
        return scope.run_id
    if lifecycle == "per_arena":
        return scope.arena_id
    return None


def _prepare_sandbox_config(config: Dict[str, Any], scope: SandboxScope) -> Dict[str, Any]:
    copied = dict(config or {})
    runtime_configs = dict(copied.get("runtime_configs") or {})
    copied["runtime_configs"] = runtime_configs
    pool_key = _build_pool_key(copied, scope)
    if pool_key:
        copied.setdefault("pool_key", pool_key)
    lifecycle = str(copied.get("lifecycle", "per_sample"))
    runtime = str(copied.get("runtime") or copied.get("backend") or "")
    if lifecycle == "per_sample" and _is_docker_runtime(runtime):
        runtime_configs.setdefault("container_name_suffix", _build_name_suffix(copied, scope))
    return copied


def _build_provider_payload(
    config: Dict[str, Any],
    scope: SandboxScope,
    pool_key: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    sandbox_id = _resolve_sandbox_id(config)
    if sandbox_id:
        payload["sandbox_id"] = sandbox_id
    payload["lifecycle"] = str(config.get("lifecycle", "per_sample"))
    if pool_key:
        payload["pool_key"] = pool_key
    scope_payload = _scope_payload(scope)
    if scope_payload:
        payload["scope"] = scope_payload
    return payload


def _scope_payload(scope: SandboxScope) -> Dict[str, str]:
    payload: Dict[str, str] = {}
    if scope.run_id:
        payload["run_id"] = scope.run_id
    if scope.task_id:
        payload["task_id"] = scope.task_id
    if scope.sample_id:
        payload["sample_id"] = scope.sample_id
    if scope.arena_id:
        payload["arena_id"] = scope.arena_id
    return payload


def _build_name_suffix(config: Dict[str, Any], scope: SandboxScope) -> str:
    sandbox_id = _resolve_sandbox_id(config)
    parts = [part for part in (sandbox_id, scope.run_id, scope.sample_id) if part]
    if not parts:
        return "sample"
    return "-".join(parts)


def _is_docker_runtime(runtime: str) -> bool:
    return runtime in {"docker", "appworld", "aio", "llm", "opensandbox"}


def _now_ms() -> float:
    import time

    return time.perf_counter() * 1000.0
