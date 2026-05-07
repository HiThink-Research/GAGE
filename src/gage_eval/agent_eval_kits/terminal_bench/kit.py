from __future__ import annotations

from gage_eval.agent_eval_kits.common import BenchmarkKitEntry
from gage_eval.agent_eval_kits.terminal_bench.config_schema import TerminalBenchKitConfig
from gage_eval.agent_eval_kits.terminal_bench.resources import build_resource_plan
from gage_eval.agent_eval_kits.terminal_bench.runtime import TerminalBenchRuntime
from gage_eval.agent_eval_kits.terminal_bench.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.terminal_bench.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.terminal_bench.tools import build_tool_registry
from gage_eval.agent_runtime.verifier.adapters import NativeVerifierAdapter


def load_kit() -> BenchmarkKitEntry:
    """Load the terminal benchmark kit entry."""

    runtime = TerminalBenchRuntime()
    return BenchmarkKitEntry(
        kit_id=runtime.benchmark_kit_id,
        config_schema=TerminalBenchKitConfig,
        default_environment_provider="docker",
        default_environment_profile_by_provider={"docker": "terminal_bench_runtime"},
        environment_profiles={
            "terminal_bench_runtime": {
                "asset_dir": "src/gage_eval/agent_eval_kits/terminal_bench/environment/docker",
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
        workflow_resolver=resolve_workflow_bundle,
        tool_registry_factory=build_tool_registry,
        verifier_adapter_factory=lambda: NativeVerifierAdapter("terminal_bench.native_verifier"),
        artifact_manifest_factory=build_artifact_manifest,
    )


def resolve_workflow_bundle(scheduler_type: str):
    """Resolve the terminal benchmark workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle()
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle()
    raise KeyError(f"Unsupported terminal_bench scheduler '{scheduler_type}'")


def build_artifact_manifest() -> dict[str, object]:
    """Declare terminal benchmark-owned artifact ids."""

    return {
        "artifacts": {
            "tool_trace": {"path": "tool_trace.json"},
            "stdout": {"path": "stdout.log"},
            "stderr": {"path": "stderr.log"},
            "workspace_diff": {"path": "workspace_diff.json"},
        }
    }


__all__ = ["build_resource_plan", "load_kit", "resolve_workflow_bundle"]
