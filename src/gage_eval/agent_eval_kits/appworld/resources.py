from __future__ import annotations


from typing import Any

from gage_eval.agent_eval_kits.common import build_environment_resource_plan


_APPWORLD_PROFILES = {
    "appworld_local": {
        "asset_dir": "src/gage_eval/agent_eval_kits/appworld/environment/docker",
        "capabilities": {
            "supports_upload_download": True,
            "supports_privileged_dind": False,
        },
    }
}


def build_resource_plan(
    runtime_spec,
    *,
    environment_profile: dict[str, Any] | None = None,
    provider_config: dict[str, Any] | None = None,
    resources: dict[str, Any] | None = None,
    startup_env: dict[str, Any] | None = None,
    lifecycle: str | None = None,
) -> dict[str, object]:
    """Build the AppWorld resource plan."""

    return build_environment_resource_plan(
        runtime_spec=runtime_spec,
        default_provider=str((environment_profile or {}).get("provider") or "docker"),
        default_profile_id="appworld_local",
        environment_profiles=_APPWORLD_PROFILES,
        environment_profile=environment_profile,
        provider_config=provider_config,
        resources=resources,
        startup_env=startup_env,
        lifecycle=lifecycle,
    )
