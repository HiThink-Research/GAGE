from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.common import BenchmarkKitEntry, build_environment_resource_plan
from gage_eval.agent_eval_kits.swebench.config_schema import SwebenchKitConfig
from gage_eval.agent_eval_kits.swebench.judge.adapters import SwebenchVerifierAdapter
from gage_eval.agent_eval_kits.swebench.runtime import SwebenchRuntime
from gage_eval.agent_eval_kits.swebench.sub_workflows.framework_loop import (
    build_workflow_bundle as build_framework_loop_bundle,
)
from gage_eval.agent_eval_kits.swebench.sub_workflows.installed_client import (
    build_workflow_bundle as build_installed_client_bundle,
)
from gage_eval.agent_eval_kits.swebench.tools import build_tool_registry


_SWEBENCH_DOCKER_PROFILE_ID = "swebench_runtime"
_SWEBENCH_E2B_PROFILE_ID = "swebench-e2b-wrapper"


def load_kit() -> BenchmarkKitEntry:
    """Load the SWE-bench kit entry."""

    runtime = SwebenchRuntime()
    return BenchmarkKitEntry(
        kit_id=runtime.benchmark_kit_id,
        config_schema=SwebenchKitConfig,
        default_environment_provider="docker",
        default_environment_profile_by_provider={
            "docker": _SWEBENCH_DOCKER_PROFILE_ID,
            "e2b": _SWEBENCH_E2B_PROFILE_ID,
        },
        environment_profiles={
            _SWEBENCH_DOCKER_PROFILE_ID: {
                "asset_dir": "src/gage_eval/agent_eval_kits/swebench/environment/docker",
                "config": {
                    "network_policy": "block",
                    "docker_platform": "linux/amd64",
                    "entrypoint": [],
                    "keepalive_command": ["sleep", "infinity"],
                    "workdir": "/workspace",
                    "exec_workdir": "/app",
                },
                "capabilities": {
                    "supports_upload_download": True,
                    "supports_privileged_dind": True,
                },
            },
            _SWEBENCH_E2B_PROFILE_ID: {
                "asset_dir": "src/gage_eval/agent_eval_kits/swebench/environment/e2b",
                "config": {
                    "template_id": "gage-swebench-pro-wrapper",
                },
                "capabilities": {
                    "supports_upload_download": True,
                    "supports_privileged_dind": False,
                },
            },
        },
        verifier_environment_policy="fresh_from_profile",
        verifier_environment_profile_id=_SWEBENCH_DOCKER_PROFILE_ID,
        supported_schedulers=runtime.supported_schedulers,
        runtime_entry=runtime,
        workflow_resolver=resolve_workflow_bundle,
        tool_registry_factory=build_tool_registry,
        verifier_adapter_factory=lambda: SwebenchVerifierAdapter(swebench_pro_mode=True),
        artifact_manifest_factory=build_artifact_manifest,
        provider_config_resolver=resolve_provider_config,
    )


def resolve_workflow_bundle(scheduler_type: str):
    """Resolve the SWE-bench workflow bundle."""

    if scheduler_type == "installed_client":
        return build_installed_client_bundle()
    if scheduler_type == "framework_loop":
        return build_framework_loop_bundle()
    raise KeyError(f"Unsupported swebench scheduler '{scheduler_type}'")


def build_resource_plan(
    runtime_spec,
    *,
    environment_profile: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    startup_env: dict[str, Any] | None = None,
    lifecycle: str | None = None,
) -> dict[str, object]:
    """Build the local-first resource plan for SWE-bench."""

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
        provider_config_resolver=resolve_provider_config,
    )


def resolve_provider_config(
    *,
    sample: dict[str, Any] | None,
    base_provider_config: dict[str, Any],
    provider: str,
    profile_id: str,
) -> dict[str, Any]:
    """Inject SWE-bench sample-specific provider config."""

    del profile_id
    provider_config = dict(base_provider_config or {})
    if provider != "docker":
        return provider_config

    metadata = sample.get("metadata") if isinstance(sample, dict) else None
    overrides = metadata.get("environment_overrides") if isinstance(metadata, dict) else None
    image_uri = overrides.get("image_uri") if isinstance(overrides, dict) else None
    if not isinstance(image_uri, str) or not image_uri.strip():
        raise ValueError("config.swebench.image_uri.missing")
    provider_config["image"] = image_uri.strip()
    return provider_config


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve SWE-bench verifier resources from the kit-owned judge adapter."""

    return {"adapter": SwebenchVerifierAdapter(swebench_pro_mode=True)}


def build_artifact_manifest() -> dict[str, object]:
    """Declare SWE-bench benchmark-owned artifact ids."""

    return {
        "artifacts": {
            "submission_patch": {"path": "submission.patch"},
            "verifier_result": {"path": "verifier/result.json"},
        }
    }


__all__ = ["build_resource_plan", "load_kit", "resolve_provider_config", "resolve_workflow_bundle"]
