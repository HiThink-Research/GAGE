from __future__ import annotations


def build_resource_plan(runtime_spec, sandbox_config: dict | None = None) -> dict[str, object]:
    """Build the Tau2 local-process resource plan."""

    effective = dict(sandbox_config or {})
    if not effective:
        effective = {
            "runtime": "tau2",
            "sandbox_id": runtime_spec.sandbox_profile_id or "tau2_local",
            "lifecycle": runtime_spec.resource_policy.get("lifecycle") or "per_sample",
            "runtime_configs": {},
        }
    return {
        "resource_kind": "local_process",
        "sandbox_config": effective,
        "cleanup_policy": {"mode": "provider_release"},
    }
