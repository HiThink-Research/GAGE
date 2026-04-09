from __future__ import annotations


def build_resource_plan(runtime_spec, sandbox_config: dict | None = None) -> dict[str, object]:
    """Build the AppWorld resource plan."""

    effective = dict(sandbox_config or {})
    if not effective:
        effective = {
            "runtime": "docker",
            "sandbox_id": runtime_spec.sandbox_profile_id or "appworld_local",
            "lifecycle": runtime_spec.resource_policy.get("lifecycle") or "per_sample",
            "runtime_configs": {},
        }
    return {
        "resource_kind": "docker",
        "sandbox_config": effective,
        "cleanup_policy": {"mode": "provider_release"},
    }
