"""Terminal benchmark resource declarations."""

from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    TERMINAL_BENCH_DEFAULT_TIMEOUT_SEC,
    TERMINAL_BENCH_REQUIRED_SURFACES,
)
def build_resource_requirements(sample: dict, plan) -> dict:
    """Return the resource requirements needed by terminal benchmark runs."""

    runtime_spec = getattr(plan, "runtime_spec", None)
    resource_policy = getattr(runtime_spec, "resource_policy", None)
    sandbox_policy = getattr(runtime_spec, "sandbox_policy", None)
    client_surface_policy = getattr(runtime_spec, "client_surface_policy", None)

    required_surfaces = tuple(
        getattr(client_surface_policy, "required", TERMINAL_BENCH_REQUIRED_SURFACES) or TERMINAL_BENCH_REQUIRED_SURFACES
    )
    optional_surfaces = tuple(getattr(client_surface_policy, "optional", ()) or ())

    timeout_sec = getattr(resource_policy, "timeout_sec", None) or TERMINAL_BENCH_DEFAULT_TIMEOUT_SEC
    environment_kind = getattr(resource_policy, "environment_kind", None) or "remote"
    prefer_remote = bool(getattr(sandbox_policy, "prefer_remote", True))
    remote_mode = getattr(sandbox_policy, "remote_mode", None) or "attached"

    return {
        "benchmark_kit_id": getattr(plan, "benchmark_kit_id", "terminal_bench"),
        "environment_kind": environment_kind,
        "required_surfaces": required_surfaces,
        "optional_surfaces": optional_surfaces,
        "timeout_sec": timeout_sec,
        "prefer_remote": prefer_remote,
        "remote_mode": remote_mode,
        "client_id": getattr(plan, "client_id", None),
        "role_adapter_id": getattr(plan, "role_adapter_id", None),
        "sample_id": sample.get("instance_id") or sample.get("sample_id") or sample.get("id"),
    }
