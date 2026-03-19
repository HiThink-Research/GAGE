"""SandboxManager orchestrates sandbox runtime creation and pooling."""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Any, Callable, Dict, Optional, Type

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.sandbox.base import BaseSandbox
from gage_eval.sandbox.docker_runtime import DockerSandbox
from gage_eval.sandbox.lease_registry import SandboxLease, SandboxLeaseRegistry
from gage_eval.sandbox.local_runtime import LocalSubprocessSandbox
from gage_eval.sandbox.pool import SandboxPool
from gage_eval.sandbox.remote_runtime import RemoteSandbox
from gage_eval.sandbox.tau2_runtime import Tau2Runtime


@dataclass
class SandboxHandle:
    """Handle representing an acquired sandbox and its metadata."""

    sandbox: BaseSandbox
    config: Dict[str, Any]
    runtime_handle: Dict[str, Any]
    pool_key: Optional[str] = None
    pool: Optional[SandboxPool] = None


class SandboxManager:
    """Factory + pool manager for sandbox runtimes."""

    def __init__(self, profiles: Optional[Dict[str, Dict[str, Any]]] = None) -> None:
        self._profiles = profiles or {}
        self._runtime_registry: Dict[str, Type[BaseSandbox]] = {
            "docker": DockerSandbox,
            "local": LocalSubprocessSandbox,
            "remote": RemoteSandbox,
        }
        self._runtime_aliases: Dict[str, str] = {
            "tau2": "local",
            "aio": "docker",
            "appworld": "docker",
            "llm": "docker",
            "opensandbox": "docker",
        }
        self._runtime_enhancers: Dict[str, Type[BaseSandbox]] = {
            "tau2": Tau2Runtime,
        }
        self._pools: Dict[str, SandboxPool] = {}
        self._pools_lock = threading.Lock()
        self._active: Dict[int, BaseSandbox] = {}
        self._active_lock = threading.Lock()
        self._lease_registry = SandboxLeaseRegistry()
        self._startup_cleanup_done = False
        self._startup_cleanup_lock = threading.Lock()
        self._shutdown = False

    def register_runtime(self, runtime: str, runtime_cls: Type[BaseSandbox]) -> None:
        self._runtime_registry[runtime] = runtime_cls

    def __del__(self) -> None:  # pragma: no cover - best-effort interpreter cleanup
        try:
            self.shutdown()
        except Exception:
            pass

    def resolve_config(
        self,
        role_config: Dict[str, Any],
        sample_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
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
        run_id: Optional[str] = None,
        task_id: Optional[str] = None,
        sample_id: Optional[str] = None,
    ) -> SandboxHandle:
        """Acquire a sandbox instance, reusing a pool when configured."""

        # STEP 1: Resolve the logical runtime into the concrete implementation.
        self._ensure_startup_cleanup(trace)
        effective = dict(config or {})
        raw_runtime = str(
            effective.get("runtime") or effective.get("backend") or "docker"
        )
        transport, runtime_cls = self._resolve_runtime(raw_runtime)

        # STEP 2: Derive pooling metadata and emit the acquire start event.
        lifecycle = effective.get("lifecycle", "per_sample")
        pool_key = effective.get("pool_key")
        if pool_key is None and lifecycle != "per_sample":
            pool_key = (
                effective.get("sandbox_id")
                or effective.get("template_name")
                or raw_runtime
            )
        payload = _build_acquire_payload(
            effective, raw_runtime, lifecycle, pool_key, transport
        )
        if trace:
            trace.emit("sandbox_acquire_start", payload, sample_id=sample_id)
        start = time.perf_counter()

        # STEP 3: Acquire an existing pooled sandbox or build a tracked runtime.
        try:
            pool = None
            if pool_key:
                pool = self._get_or_create_pool(
                    pool_key=pool_key,
                    builder=lambda _cls=runtime_cls, _cfg=effective, _runtime=raw_runtime, _pool_key=pool_key, **kw: self._build_tracked_sandbox(
                        runtime_cls=_cls,
                        runtime=_runtime,
                        config=_cfg,
                        pool_key=_pool_key,
                        trace=kw.get("trace"),
                        run_id=kw.get("run_id"),
                        task_id=kw.get("task_id"),
                        sample_id=None,
                    ),
                    max_size=effective.get("pool_max") or effective.get("pool_size"),
                    max_uses=effective.get("max_uses")
                    or effective.get("max_container_uses"),
                    idle_timeout_s=effective.get("idle_timeout_s"),
                )
            else:
                self._ensure_not_shutdown()
            sandbox = (
                pool.acquire(
                    trace=trace,
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                )
                if pool
                else self._build_tracked_sandbox(
                    runtime_cls=runtime_cls,
                    runtime=raw_runtime,
                    config=effective,
                    pool_key=pool_key,
                    trace=trace,
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                )
            )
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

        # STEP 4: Emit the acquire completion event and return the handle.
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
        return SandboxHandle(
            sandbox=sandbox,
            config=effective,
            runtime_handle=runtime_handle,
            pool_key=pool_key,
            pool=pool,
        )

    def release(self, handle: SandboxHandle) -> None:
        """Release a sandbox handle back to its pool or tear it down."""

        if handle.pool is not None:
            handle.pool.release(handle.sandbox)
            return
        self._unregister_active(handle.sandbox)
        handle.sandbox.teardown()

    def shutdown(self) -> None:
        """Shut down all managed pools and active sandbox instances."""

        with self._pools_lock:
            self._shutdown = True
            pools = list(self._pools.values())
            self._pools.clear()
        for pool in pools:
            pool.shutdown()
        active = self._drain_active()
        for sandbox in active:
            try:
                sandbox.teardown()
            except Exception:
                pass

    def close(self) -> None:
        """Close all managed pools and active sandbox instances."""

        self.shutdown()

    def _resolve_runtime(self, runtime: str) -> tuple[str, Type[BaseSandbox]]:
        transport = self._runtime_aliases.get(runtime, runtime)
        runtime_cls = self._runtime_enhancers.get(runtime) or self._runtime_registry.get(
            transport
        )
        if runtime_cls is None:
            raise KeyError(
                f"Unknown sandbox runtime '{runtime}' (resolved transport '{transport}')"
            )
        return transport, runtime_cls

    def _get_or_create_pool(
        self,
        *,
        pool_key: str,
        builder: Callable[..., BaseSandbox],
        max_size: Optional[int],
        max_uses: Optional[int],
        idle_timeout_s: Optional[float],
    ) -> SandboxPool:
        with self._pools_lock:
            if self._shutdown:
                raise RuntimeError("SandboxManager is shut down")
            pool = self._pools.get(pool_key)
            if pool is None:
                pool = SandboxPool(
                    builder=builder,
                    max_size=max_size,
                    max_uses=max_uses,
                    idle_timeout_s=idle_timeout_s,
                )
                self._pools[pool_key] = pool
            return pool

    def _ensure_not_shutdown(self) -> None:
        with self._pools_lock:
            if self._shutdown:
                raise RuntimeError("SandboxManager is shut down")

    def _ensure_startup_cleanup(self, trace: Optional[ObservabilityTrace]) -> None:
        if self._startup_cleanup_done:
            return
        with self._startup_cleanup_lock:
            if self._startup_cleanup_done:
                return
            leases = list(self._lease_registry.iter_leases())
            if trace:
                trace.emit("sandbox_stale_cleanup_scan", {"lease_count": len(leases)})
            for lease in leases:
                if not self._lease_registry.is_stale(lease):
                    continue
                self._cleanup_stale_lease(lease, trace)
            self._startup_cleanup_done = True

    def _cleanup_stale_lease(
        self,
        lease: SandboxLease,
        trace: Optional[ObservabilityTrace],
    ) -> None:
        payload = _build_stale_cleanup_payload(lease)
        if trace:
            trace.emit("sandbox_stale_cleanup_start", payload)
        try:
            _, runtime_cls = self._resolve_runtime(lease.runtime)
        except KeyError:
            if trace:
                failed = dict(payload)
                failed.update({"status": "failed", "error": f"unknown_runtime:{lease.runtime}"})
                trace.emit("sandbox_stale_cleanup_end", failed)
            return
        start = time.perf_counter()
        try:
            cleaned = runtime_cls.cleanup_stale_runtime(lease.config, lease.runtime_handle)
        except Exception as exc:
            if trace:
                failed = dict(payload)
                failed.update(
                    {
                        "status": "failed",
                        "error": str(exc),
                        "latency_ms": (time.perf_counter() - start) * 1000.0,
                    }
                )
                trace.emit("sandbox_stale_cleanup_end", failed)
            return
        if cleaned:
            self._lease_registry.release(lease.lease_id)
        if trace:
            finished = dict(payload)
            finished.update(
                {
                    "status": "success" if cleaned else "skipped",
                    "latency_ms": (time.perf_counter() - start) * 1000.0,
                }
            )
            trace.emit("sandbox_stale_cleanup_end", finished)

    def _build_tracked_sandbox(
        self,
        *,
        runtime_cls: Type[BaseSandbox],
        runtime: str,
        config: Dict[str, Any],
        pool_key: Optional[str],
        trace: Optional[ObservabilityTrace],
        run_id: Optional[str],
        task_id: Optional[str],
        sample_id: Optional[str],
    ) -> BaseSandbox:
        # STEP 1: Attach management metadata before the runtime starts.
        tracked_config = _with_management_metadata(
            config,
            runtime=runtime,
            pool_key=pool_key,
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
        )
        sandbox = self._build_sandbox(runtime_cls, tracked_config, trace, sample_id)

        # STEP 2: Install teardown tracking so leases survive pool release but disappear on teardown.
        self._install_teardown_tracking(sandbox)

        # STEP 3: Persist a lease for crash-time orphan recovery.
        runtime_handle = getattr(sandbox, "_runtime_handle", {}) or {}
        try:
            lease = self._lease_registry.register(
                runtime=runtime,
                sandbox_id=_resolve_sandbox_id(tracked_config),
                pool_key=pool_key,
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                config=tracked_config,
                runtime_handle=runtime_handle,
            )
        except Exception:
            sandbox.teardown()
            raise
        setattr(sandbox, "_gage_lease_id", lease.lease_id)
        return sandbox

    def _install_teardown_tracking(self, sandbox: BaseSandbox) -> None:
        if getattr(sandbox, "_gage_teardown_tracking", False):
            return
        original_teardown: Callable[[], None] = sandbox.teardown

        def tracked_teardown() -> None:
            original_teardown()
            lease_id = getattr(sandbox, "_gage_lease_id", None)
            if lease_id:
                self._lease_registry.release(str(lease_id))
                setattr(sandbox, "_gage_lease_id", None)

        setattr(sandbox, "teardown", tracked_teardown)
        setattr(sandbox, "_gage_teardown_tracking", True)

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
    transport: Optional[str] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "runtime": runtime,
        "lifecycle": str(lifecycle),
    }
    if transport and transport != runtime:
        payload["transport"] = transport
    sandbox_id = config.get("sandbox_id") or config.get("template_name")
    if sandbox_id:
        payload["sandbox_id"] = sandbox_id
    if pool_key:
        payload["pool_key"] = pool_key
    return payload


def _resolve_sandbox_id(config: Dict[str, Any]) -> Optional[str]:
    sandbox_id = config.get("sandbox_id") or config.get("template_name")
    if sandbox_id:
        return str(sandbox_id)
    return None


def _with_management_metadata(
    config: Dict[str, Any],
    *,
    runtime: str,
    pool_key: Optional[str],
    run_id: Optional[str],
    task_id: Optional[str],
    sample_id: Optional[str],
) -> Dict[str, Any]:
    copied = dict(config or {})
    metadata = {
        "managed": True,
        "runtime": str(runtime),
        "owner_pid": os.getpid(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    sandbox_id = _resolve_sandbox_id(copied)
    if sandbox_id:
        metadata["sandbox_id"] = sandbox_id
    if pool_key:
        metadata["pool_key"] = str(pool_key)
    if run_id:
        metadata["run_id"] = str(run_id)
    if task_id:
        metadata["task_id"] = str(task_id)
    if sample_id:
        metadata["sample_id"] = str(sample_id)
    copied["_gage_managed_metadata"] = metadata
    return copied


def _build_runtime_payload(config: Dict[str, Any]) -> Dict[str, Any]:
    payload: Dict[str, Any] = {}
    runtime = config.get("runtime") or config.get("backend")
    if runtime:
        payload["runtime"] = runtime
    sandbox_id = _resolve_sandbox_id(config)
    if sandbox_id:
        payload["sandbox_id"] = sandbox_id
    return payload


def _inject_runtime_handle(
    payload: Dict[str, Any], runtime_handle: Dict[str, Any]
) -> None:
    if not runtime_handle:
        return
    for key in ("container_id", "container_name"):
        value = runtime_handle.get(key)
        if value:
            payload[key] = value
    for key in (
        "env_endpoint",
        "environment_endpoint",
        "apis_endpoint",
        "mcp_endpoint",
    ):
        value = runtime_handle.get(key)
        if value:
            payload[key] = value


def _build_stale_cleanup_payload(lease: SandboxLease) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "lease_id": lease.lease_id,
        "runtime": lease.runtime,
        "owner_pid": lease.owner_pid,
        "owner_host": lease.owner_host,
    }
    if lease.sandbox_id:
        payload["sandbox_id"] = lease.sandbox_id
    if lease.pool_key:
        payload["pool_key"] = lease.pool_key
    if lease.run_id:
        payload["run_id"] = lease.run_id
    if lease.task_id:
        payload["task_id"] = lease.task_id
    if lease.sample_id:
        payload["sample_id"] = lease.sample_id
    _inject_runtime_handle(payload, lease.runtime_handle)
    return payload
