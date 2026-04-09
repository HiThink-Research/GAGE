from __future__ import annotations

from gage_eval.agent_eval_kits.common import BenchmarkKitEntry
from gage_eval.agent_eval_kits.tau2.judge_bridge import resolve_verifier_resources
from gage_eval.agent_eval_kits.tau2.legacy_support_migration import resolve_compat_shim
from gage_eval.agent_eval_kits.tau2.resources import build_resource_plan
from gage_eval.agent_eval_kits.tau2.runtime import Tau2RuntimeEntry
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.tau2.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.tau2.trace_mapping import map_trace_payload


def load_kit() -> BenchmarkKitEntry:
    """Load the Tau2 kit entry."""

    runtime = Tau2RuntimeEntry()
    return BenchmarkKitEntry(
        benchmark_kit_id=runtime.benchmark_kit_id,
        runtime_version=runtime.runtime_version,
        supported_schedulers=runtime.supported_schedulers,
        verifier_kind=runtime.verifier_kind,
        resource_requirements=runtime.resource_requirements,
        lifecycle_policy=runtime.lifecycle_policy,
        state_schema_keys=runtime.state_schema_keys,
        compat_mode=runtime.compat_mode,
        runtime_entry=runtime,
        workflow_resolver=resolve_workflow_bundle,
        verifier_resource_resolver=resolve_verifier_resources,
        trace_mapper=map_trace_payload,
        compat_shim_resolver=resolve_compat_shim,
    )


def resolve_workflow_bundle(scheduler_type: str):
    """Resolve the Tau2 workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle()
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle()
    raise KeyError(f"Unsupported tau2 scheduler '{scheduler_type}'")


__all__ = ["build_resource_plan", "load_kit", "resolve_workflow_bundle"]
