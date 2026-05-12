"""Bridge GAGE environment specs to Harbor trial environment configs."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Mapping

from gage_eval.external_harness_kits.errors import ExternalHarnessError

_SUPPORTED_PROVIDERS = {"docker", "e2b"}
_SECRET_KEY_PARTS = ("secret", "token", "api_key", "apikey", "password", "credential")


@dataclass(frozen=True)
class HarborEnvironmentBinding:
    env_id: str
    provider: str
    harbor_environment: dict[str, Any]
    preflight_notes: list[str]


def build_harbor_environment_binding(
    environment: Mapping[str, Any],
    *,
    environment_override: Mapping[str, Any] | None = None,
    validation_phase: str = "config",
) -> HarborEnvironmentBinding:
    """Translate a GAGE environment into a Harbor trial sandbox binding."""

    env_id = str(environment.get("env_id") or "")
    provider = str(environment.get("provider") or "")
    if provider == "local_process":
        raise ExternalHarnessError(
            "external_harness.config.invalid_environment_provider",
            f"environment '{env_id}' provider local_process is not a Harbor trial sandbox provider",
        )
    if provider not in _SUPPORTED_PROVIDERS:
        raise ExternalHarnessError(
            "external_harness.config.invalid_environment_provider",
            f"environment '{env_id}' provider '{provider}' is unsupported for Harbor trial sandbox",
        )

    translated: dict[str, Any] = {"type": provider}
    _apply_resources(translated, _mapping(environment.get("resources")))
    provider_config = _mapping(environment.get("provider_config"))
    notes: list[str] = []
    if provider == "docker":
        translated["delete"] = False
        _apply_docker_provider_config(translated, provider_config, env_id=env_id)
    elif provider == "e2b":
        _apply_e2b_provider_config(translated, provider_config, env_id=env_id)

    if environment_override:
        translated = _merge_environment_override(
            base=translated,
            override=_mapping(environment_override),
            env_id=env_id,
            validation_phase=validation_phase,
        )
    if provider == "e2b":
        notes.extend(_e2b_deferred_live_notes(translated, provider_config))
    return HarborEnvironmentBinding(
        env_id=env_id,
        provider=provider,
        harbor_environment=_strip_empty_containers(translated),
        preflight_notes=notes,
    )


def _apply_resources(target: dict[str, Any], resources: Mapping[str, Any]) -> None:
    key_map = {
        "cpus": "override_cpus",
        "cpu": "override_cpus",
        "memory_mb": "override_memory_mb",
        "storage_mb": "override_storage_mb",
        "disk_mb": "override_storage_mb",
        "gpus": "override_gpus",
    }
    for source_key, target_key in key_map.items():
        if source_key in resources and resources[source_key] is not None:
            target[target_key] = resources[source_key]


def _apply_docker_provider_config(
    target: dict[str, Any],
    provider_config: Mapping[str, Any],
    *,
    env_id: str,
) -> None:
    kwargs = _mapping(provider_config.get("kwargs"))
    if provider_config.get("docker_image") is not None:
        kwargs = {**kwargs, "docker_image": provider_config["docker_image"]}
    if kwargs:
        target["kwargs"] = dict(kwargs)
    if provider_config.get("mounts_json") is not None:
        target["mounts_json"] = provider_config["mounts_json"]
    if provider_config.get("env") is not None:
        target["env"] = _non_secret_env(provider_config["env"], env_id=env_id)


def _apply_e2b_provider_config(
    target: dict[str, Any],
    provider_config: Mapping[str, Any],
    *,
    env_id: str,
) -> None:
    kwargs = _mapping(provider_config.get("kwargs"))
    for key in ("template_id", "timeout_s", "request_timeout_s"):
        if provider_config.get(key) is not None:
            kwargs = {**kwargs, key: provider_config[key]}
    if kwargs:
        target["kwargs"] = dict(kwargs)
    if provider_config.get("env") is not None:
        target["env"] = _non_secret_env(provider_config["env"], env_id=env_id)


def _e2b_deferred_live_notes(
    harbor_environment: Mapping[str, Any],
    provider_config: Mapping[str, Any],
) -> list[str]:
    notes: list[str] = []
    kwargs = _mapping(harbor_environment.get("kwargs"))
    template_id = kwargs.get("template_id")
    if not template_id:
        notes.append("deferred live check: E2B template id is not configured")
    token = (
        provider_config.get("api_key")
        or provider_config.get("token")
        or provider_config.get("e2b_api_key")
        or os.environ.get("E2B_API_KEY")
    )
    if not token:
        notes.append("deferred live check: E2B token is not configured")
    return notes


def _merge_environment_override(
    *,
    base: dict[str, Any],
    override: Mapping[str, Any],
    env_id: str,
    validation_phase: str,
) -> dict[str, Any]:
    override_type = override.get("type")
    if override_type is not None and override_type != base.get("type"):
        raise ExternalHarnessError(
            _provider_mismatch_code(validation_phase),
            f"environment '{env_id}' override type '{override_type}' conflicts with provider-derived type '{base.get('type')}'",
        )
    if override.get("import_path") is not None:
        raise ExternalHarnessError(
            "external_harness.config.invalid_environment_override",
            f"environment '{env_id}' environment_override.import_path is not allowed for provider-derived Harbor environments",
        )
    sanitized = dict(override)
    sanitized.pop("type", None)
    if "env" in sanitized:
        sanitized["env"] = _non_secret_env(sanitized["env"], env_id=env_id)
    return _deep_merge(base, sanitized)


def _provider_mismatch_code(validation_phase: str) -> str:
    if str(validation_phase).strip().lower() == "environment":
        return "external_harness.environment.provider_mismatch"
    return "external_harness.config.provider_mismatch"


def _deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = _deep_merge(dict(result[key]), value)
        else:
            result[key] = value
    return result


def _non_secret_env(value: Any, *, env_id: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ExternalHarnessError(
            "external_harness.config.invalid_environment_override",
            f"environment '{env_id}' provider_config.env must be a mapping",
        )
    normalized: dict[str, str] = {}
    for key, env_value in value.items():
        key_text = str(key)
        if _looks_secret_like(key_text):
            raise ExternalHarnessError(
                "external_harness.config.invalid_environment_override",
                f"environment '{env_id}' provider_config.env contains secret-like key '{key_text}'",
            )
        normalized[key_text] = str(env_value)
    return normalized


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _looks_secret_like(value: str) -> bool:
    lowered = value.lower()
    return any(part in lowered for part in _SECRET_KEY_PARTS)


def _strip_empty_containers(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in payload.items()
        if value not in ({}, [], None)
    }


__all__ = ["HarborEnvironmentBinding", "build_harbor_environment_binding"]
