from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.judge_bridge import resolve_verifier_resources
from gage_eval.agent_eval_kits.appworld.resources import build_resource_plan
from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime
from gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.appworld.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.appworld.trace_mapping import map_trace_payload
from gage_eval.agent_eval_kits.common import BenchmarkKitEntry


def load_kit() -> BenchmarkKitEntry:
    """Load the AppWorld kit entry."""

    runtime = AppWorldRuntime()
    return BenchmarkKitEntry(
        benchmark_kit_id=runtime.benchmark_kit_id,
        runtime_version=runtime.runtime_version,
        supported_schedulers=runtime.supported_schedulers,
        verifier_kind=runtime.verifier_kind,
        resource_requirements=runtime.resource_requirements,
        lifecycle_policy=runtime.lifecycle_policy,
        state_schema_keys=runtime.state_schema_keys,
        runtime_entry=runtime,
        workflow_resolver=lambda scheduler_type: resolve_workflow_bundle(runtime, scheduler_type),
        verifier_resource_resolver=resolve_verifier_resources,
        trace_mapper=map_trace_payload,
    )


def resolve_workflow_bundle(runtime: AppWorldRuntime, scheduler_type: str):
    """Resolve the AppWorld workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle(runtime)
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle(runtime)
    raise KeyError(f"Unsupported appworld scheduler '{scheduler_type}'")


__all__ = ["build_resource_plan", "load_kit", "resolve_workflow_bundle"]
