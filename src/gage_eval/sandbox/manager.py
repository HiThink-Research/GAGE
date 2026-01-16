"""SandboxManager orchestrates sandbox runtime creation and pooling."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional, Type

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.sandbox.aio_runtime import AioSandbox
from gage_eval.sandbox.appworld_runtime import AppWorldRuntime
from gage_eval.sandbox.base import BaseSandbox
from gage_eval.sandbox.docker_runtime import DockerSandbox
from gage_eval.sandbox.llm_runtime import LlmSandbox
from gage_eval.sandbox.local_runtime import LocalSubprocessSandbox
from gage_eval.sandbox.opensandbox_runtime import OpenSandbox
from gage_eval.sandbox.pool import SandboxPool
from gage_eval.sandbox.remote_runtime import RemoteSandbox


@dataclass
class SandboxHandle:
    """Handle representing an acquired sandbox and its metadata."""

    sandbox: BaseSandbox
    config: Dict[str, Any]
    runtime_handle: Dict[str, Any]
    pool_key: Optional[str] = None


class SandboxManager:
    """Factory + pool manager for sandbox runtimes."""

    def __init__(self, profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._profiles = profiles or {}
        self._runtime_registry: Dict[str, Type[BaseSandbox]] = {
            "docker": DockerSandbox,
            "local": LocalSubprocessSandbox,
            "remote": RemoteSandbox,
            "aio": AioSandbox,
            "appworld": AppWorldRuntime,
            "llm": LlmSandbox,
            "opensandbox": OpenSandbox,
        }
        self._pools: Dict[str, SandboxPool] = {}
        self._active: Dict[int, BaseSandbox] = {}
        self._active_lock = threading.Lock()

    def register_runtime(self, runtime: str, runtime_cls: Type[BaseSandbox]) -> None:
        self._runtime_registry[runtime] = runtime_cls

    def resolve_config(self, role_config: Dict[str, Any], sample_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Merge sandbox config with optional template and sample overrides."""

        base = dict(role_config or {})
        sandbox_id = base.get("sandbox_id") or base.get("template_name")
        if sandbox_id and sandbox_id in self._profiles:
            base = _deep_merge(self._profiles[sandbox_id], base)
        if sample_config:
            base = _deep_merge(base, sample_config)
        return base

    def acquire(
        self,
        config: Dict[str, Any],
        *,
        trace: Optional[ObservabilityTrace] = None,
        sample_id: Optional[str] = None,
    ) -> SandboxHandle:
        effective = dict(config or {})
        runtime = effective.get("runtime") or effective.get("backend") or "docker"
        runtime_cls = self._runtime_registry.get(runtime)
        if runtime_cls is None:
            raise KeyError(f"Unknown sandbox runtime '{runtime}'")
        lifecycle = effective.get("lifecycle", "per_sample")
        pool_key = effective.get("pool_key")
        if pool_key is None and lifecycle != "per_sample":
            pool_key = effective.get("sandbox_id") or effective.get("template_name") or runtime
        payload = _build_acquire_payload(effective, runtime, lifecycle, pool_key)
        if trace:
            trace.emit("sandbox_acquire_start", payload, sample_id=sample_id)
        start = time.perf_counter()
        try:
            pool = None
            if pool_key:
                pool = self._pools.setdefault(
                    pool_key,
                    SandboxPool(
                        builder=lambda: self._build_sandbox(runtime_cls, effective, trace, sample_id),
                        max_size=effective.get("pool_max") or effective.get("pool_size"),
                        max_uses=effective.get("max_container_uses"),
                    ),
                )
            sandbox = pool.acquire() if pool else self._build_sandbox(runtime_cls, effective, trace, sample_id)
            if not pool:
                self._register_active(sandbox)
            runtime_handle = getattr(sandbox, "_runtime_handle", {}) or {}
        except Exception as exc:
            if trace:
                failure = dict(payload)
                failure.update(
                    {
                        "status": "failed",
                        "latency_ms": (time.perf_counter() - start) * 1000.0,
                        "error": str(exc),
                    }
                )
                trace.emit("sandbox_acquire_end", failure, sample_id=sample_id)
            raise
        if trace:
            success = dict(payload)
            success.update(
                {
                    "status": "success",
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
            _inject_runtime_handle(success, runtime_handle)
            trace.emit("sandbox_acquire_end", success, sample_id=sample_id)
        return SandboxHandle(sandbox=sandbox, config=effective, runtime_handle=runtime_handle, pool_key=pool_key)

    def release(self, handle: SandboxHandle) -> None:
        if handle.pool_key and handle.pool_key in self._pools:
            self._pools[handle.pool_key].release(handle.sandbox)
            return
        self._unregister_active(handle.sandbox)
        handle.sandbox.teardown()

    def shutdown(self) -> None:
        for pool in self._pools.values():
            pool.shutdown()
        self._pools.clear()
        active = self._drain_active()
        for sandbox in active:
            try:
                sandbox.teardown()
            except Exception:
                pass

    @staticmethod
    def _build_sandbox(
        runtime_cls: Type[BaseSandbox],
        config: Dict[str, Any],
        trace: Optional[ObservabilityTrace] = None,
        sample_id: Optional[str] = None,
    ) -> BaseSandbox:
        sandbox = runtime_cls(
            runtime_configs=config.get("runtime_configs"),
            resources=config.get("resources"),
        )
        payload = _build_runtime_payload(config)
        if trace:
            trace.emit("sandbox_runtime_start", payload, sample_id=sample_id)
        try:
            runtime_handle = sandbox.start(config) or {}
        except Exception:
            sandbox.teardown()
            raise
        if trace:
            ready = dict(payload)
            ready.update({"status": "success"})
            _inject_runtime_handle(ready, runtime_handle)
            trace.emit("sandbox_runtime_ready", ready, sample_id=sample_id)
        setattr(sandbox, "_runtime_handle", runtime_handle)
        return sandbox

    def _register_active(self, sandbox: BaseSandbox) -> None:
        with self._active_lock:
            self._active[id(sandbox)] = sandbox

    def _unregister_active(self, sandbox: BaseSandbox) -> None:
        with self._active_lock:
            self._active.pop(id(sandbox), None)

    def _drain_active(self) -> list[BaseSandbox]:
        with self._active_lock:
            active = list(self._active.values())
            self._active.clear()
            return active


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _build_acquire_payload(
    config: Dict[str, Any],
    runtime: str,
    lifecycle: str,
    pool_key: Optional[str],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "runtime": runtime,
        "lifecycle": str(lifecycle),
    }
    sandbox_id = config.get("sandbox_id") or config.get("template_name")
    if sandbox_id:
        payload["sandbox_id"] = sandbox_id
    if pool_key:
        payload["pool_key"] = pool_key
    return payload


def _build_runtime_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    runtime = config.get("runtime") or config.get("backend")
    if runtime:
        payload["runtime"] = runtime
    sandbox_id = config.get("sandbox_id") or config.get("template_name")
    if sandbox_id:
        payload["sandbox_id"] = sandbox_id
    return payload


def _inject_runtime_handle(payload: Dict[str, Any], runtime_handle: Dict[str, Any]) -> None:
    if not runtime_handle:
        return
    for key in ("container_id", "container_name"):
        value = runtime_handle.get(key)
        if value:
            payload[key] = value
    for key in ("env_endpoint", "environment_endpoint", "apis_endpoint", "mcp_endpoint"):
        value = runtime_handle.get(key)
        if value:
            payload[key] = value
