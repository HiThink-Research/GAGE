from __future__ import annotations

from typing import Any


def build_resource_plan(runtime_spec, sandbox_config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build the local-first resource plan for terminal benchmark."""

    effective = dict(sandbox_config or {})
    if not effective:
        effective = {
            "runtime": "docker",
            "sandbox_id": runtime_spec.sandbox_profile_id or "terminal_bench_runtime",
            "lifecycle": runtime_spec.resource_policy.get("lifecycle") or "per_sample",
            "runtime_configs": {},
        }
    return {
        "resource_kind": "docker",
        "sandbox_config": effective,
        "cleanup_policy": {"mode": "provider_release"},
    }
