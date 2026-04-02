"""AppWorld resource declarations."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.appworld.kit import build_kit

def build_resource_requirements(sample: dict, plan) -> dict:
    """Return the resource requirements needed by AppWorld runs."""

    kit = build_kit()
    runtime_spec = getattr(plan, "runtime_spec", None)
    resource_policy = getattr(runtime_spec, "resource_policy", None)
    sandbox_policy = getattr(runtime_spec, "sandbox_policy", None)
    sample_id = sample.get("task_id") or sample.get("id")
    if isinstance(sample.get("metadata"), dict):
        appworld_meta = sample["metadata"].get("appworld")
        if isinstance(appworld_meta, dict):
            sample_id = appworld_meta.get("task_id") or sample_id

    payload: Dict[str, Any] = {
        "benchmark_kit_id": getattr(plan, "benchmark_kit_id", kit.kit_id),
        "environment_kind": getattr(resource_policy, "environment_kind", None) or "docker",
        "required_surfaces": tuple(getattr(plan, "required_surfaces", ()) or kit.required_surfaces),
        "optional_surfaces": tuple(getattr(plan, "optional_surfaces", ()) or kit.optional_surfaces),
        "timeout_sec": getattr(resource_policy, "timeout_sec", None) or 1800,
        "prefer_remote": bool(getattr(sandbox_policy, "prefer_remote", False)),
        "remote_mode": getattr(sandbox_policy, "remote_mode", None),
        "sandbox_profile_id": getattr(plan, "sandbox_profile_id", None),
        "client_id": getattr(plan, "client_id", None),
        "role_adapter_id": getattr(plan, "role_adapter_id", None),
        "sample_id": str(sample_id or "unknown"),
    }
    return payload
