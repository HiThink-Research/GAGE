from __future__ import annotations

from gage_eval.agent_eval_kits.common import BenchmarkKitEntry, build_environment_resource_plan
from gage_eval.agent_eval_kits.tau2.config_schema import Tau2KitConfig
from gage_eval.agent_eval_kits.tau2.judge.adapters import Tau2VerifierAdapter
from gage_eval.agent_eval_kits.tau2.runtime import Tau2RuntimeEntry
from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.tau2.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.tau2.tools import build_tool_registry


_TAU2_LOCAL_PROCESS_PROFILE_ID = "tau2-local-process"
_TAU2_E2B_PROFILE_ID = "tau2-e2b-wrapper"


def load_kit() -> BenchmarkKitEntry:
    """Load the Tau2 kit entry."""

    runtime = Tau2RuntimeEntry()
    return BenchmarkKitEntry(
        kit_id=runtime.benchmark_kit_id,
        config_schema=Tau2KitConfig,
        default_environment_provider="local_process",
        default_environment_profile_by_provider={
            "local_process": _TAU2_LOCAL_PROCESS_PROFILE_ID,
            "e2b": _TAU2_E2B_PROFILE_ID,
        },
        environment_profiles={
            _TAU2_LOCAL_PROCESS_PROFILE_ID: {
                "asset_dir": "src/gage_eval/agent_eval_kits/tau2/environment/local_process",
                "capabilities": {
                    "supports_upload_download": True,
                    "supports_privileged_dind": False,
                },
            },
            _TAU2_E2B_PROFILE_ID: {
                "asset_dir": "src/gage_eval/agent_eval_kits/tau2/environment/e2b",
                "config": {
                    "template_id": "gage-tau2-wrapper",
                },
                "capabilities": {
                    "supports_upload_download": True,
                    "supports_privileged_dind": False,
                },
            },
        },
        verifier_environment_policy="reuse",
        verifier_environment_profile_id=None,
        supported_schedulers=runtime.supported_schedulers,
        runtime_entry=runtime,
        workflow_resolver=resolve_workflow_bundle,
        tool_registry_factory=build_tool_registry,
        verifier_adapter_factory=Tau2VerifierAdapter,
        artifact_manifest_factory=build_artifact_manifest,
    )


def resolve_workflow_bundle(scheduler_type: str):
    """Resolve the Tau2 workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle()
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle()
    raise KeyError(f"Unsupported tau2 scheduler '{scheduler_type}'")


def build_resource_plan(
    runtime_spec,
    *,
    environment_profile: dict | None = None,
    provider_config: dict | None = None,
    resources: dict | None = None,
    startup_env: dict | None = None,
    lifecycle: str | None = None,
) -> dict[str, object]:
    """Build the Tau2 local-process resource plan."""

    kit = load_kit()
    default_provider = str((environment_profile or {}).get("provider") or kit.default_environment_provider)
    default_profile_id = kit.default_environment_profile_by_provider.get(
        default_provider,
        kit.default_environment_profile_by_provider[kit.default_environment_provider],
    )
    return build_environment_resource_plan(
        runtime_spec=runtime_spec,
        default_provider=default_provider,
        default_profile_id=default_profile_id,
        environment_profiles=kit.environment_profiles,
        environment_profile=environment_profile,
        provider_config=provider_config,
        resources=resources,
        startup_env=startup_env,
        lifecycle=lifecycle,
    )


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve Tau2 verifier resources from the kit-owned judge adapter."""

    return {"adapter": Tau2VerifierAdapter()}


def build_artifact_manifest() -> dict[str, object]:
    """Declare Tau2 benchmark-owned artifact ids."""

    return {
        "artifacts": {
            "tau2_state": {"path": "tau2_state.json"},
            "tau2_trajectory": {"path": "tau2_trajectory.json"},
            "tau2_cost": {"path": "tau2_cost.json"},
        }
    }


__all__ = ["build_resource_plan", "load_kit", "resolve_workflow_bundle"]
