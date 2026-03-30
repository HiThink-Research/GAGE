"""SWE-bench resource requirements."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.agent_eval_kits.swebench.kit import build_kit


def build_resource_requirements(sample: dict, plan: Any) -> Dict[str, Any]:
    """Declare the minimal resources required by SWE-bench."""
    kit = build_kit()
    sandbox_profile_id = getattr(plan, "sandbox_profile_id", None)
    if isinstance(plan, dict):
        sandbox_profile_id = plan.get("sandbox_profile_id", sandbox_profile_id)
    return {
        "benchmark_kit_id": kit.kit_id,
        "required_surfaces": list(kit.required_surfaces),
        "optional_surfaces": list(kit.optional_surfaces),
        "sandbox_profile_id": sandbox_profile_id,
        "sample_id": sample.get("instance_id") or sample.get("id"),
    }

