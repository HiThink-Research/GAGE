from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.config_schema import AppWorldKitConfig
from gage_eval.agent_eval_kits.appworld.judge.adapters import AppWorldVerifierAdapter
from gage_eval.agent_eval_kits.appworld.resources import build_resource_plan
from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime
from gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.appworld.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.appworld.tools import build_tool_registry
from gage_eval.agent_eval_kits.common import BenchmarkKitEntry


def load_kit() -> BenchmarkKitEntry:
    """Load the AppWorld kit entry."""

    runtime = AppWorldRuntime()
    return BenchmarkKitEntry(
        kit_id=runtime.benchmark_kit_id,
        config_schema=AppWorldKitConfig,
        default_environment_provider="docker",
        default_environment_profile_by_provider={"docker": "appworld_local"},
        environment_profiles={
            "appworld_local": {
                "asset_dir": "src/gage_eval/agent_eval_kits/appworld/environment/docker",
                "capabilities": {
                    "supports_upload_download": True,
                    "supports_privileged_dind": False,
                },
            }
        },
        verifier_environment_policy="reuse",
        verifier_environment_profile_id=None,
        supported_schedulers=runtime.supported_schedulers,
        runtime_entry=runtime,
        workflow_resolver=lambda scheduler_type: resolve_workflow_bundle(runtime, scheduler_type),
        tool_registry_factory=build_tool_registry,
        verifier_adapter_factory=AppWorldVerifierAdapter,
        artifact_manifest_factory=build_artifact_manifest,
    )


def resolve_workflow_bundle(runtime: AppWorldRuntime, scheduler_type: str):
    """Resolve the AppWorld workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle(runtime)
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle(runtime)
    raise KeyError(f"Unsupported appworld scheduler '{scheduler_type}'")


def build_artifact_manifest() -> dict[str, object]:
    """Declare AppWorld benchmark-owned artifact ids."""

    return {
        "artifacts": {
            "appworld_save": {"path": "appworld_save.json"},
            "appworld_outputs": {"path": "appworld_outputs.json"},
            "appworld_tool_trace": {"path": "appworld_tool_trace.json"},
            "appworld_logs": {"path": "appworld_logs.json"},
        }
    }


__all__ = ["build_resource_plan", "load_kit", "resolve_workflow_bundle"]
