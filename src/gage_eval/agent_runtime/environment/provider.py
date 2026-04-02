"""Environment construction helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from gage_eval.agent_runtime.environment.base import AgentEnvironment
from gage_eval.agent_runtime.environment.docker_environment import DockerEnvironment
from gage_eval.agent_runtime.environment.fake import FakeEnvironment
from gage_eval.agent_runtime.environment.remote_environment import RemoteEnvironment
from gage_eval.agent_runtime.resources.sandbox_policy import resolve_sandbox_policy
from gage_eval.sandbox.contracts import (
    RemoteSandboxContract,
    deep_merge_dicts,
    merge_sandbox_profile_layers,
)


class EnvironmentProvider:
    """Phase 1 main-path factory."""

    def __init__(self, *, profiles: Optional[Mapping[str, Mapping[str, Any]]] = None) -> None:
        self._profiles = {
            str(key): dict(value)
            for key, value in (profiles or {}).items()
            if isinstance(value, Mapping)
        }

    def build(self, plan, sample) -> AgentEnvironment:
        """Construct an environment instance for the compiled plan."""

        sample_payload = _as_mapping(sample)
        runtime_spec = getattr(plan, "runtime_spec", None)
        resource_policy = getattr(runtime_spec, "resource_policy", None)
        sandbox_policy = getattr(runtime_spec, "sandbox_policy", None)
        if sandbox_policy is not None:
            sandbox_policy = resolve_sandbox_policy(sandbox_policy, sample_payload)

        runtime_configs = _build_runtime_configs(runtime_spec, resource_policy, sample_payload)
        resources = _build_resources(resource_policy, sample_payload)
        environment_kind = getattr(plan, "environment_kind", None) or getattr(
            resource_policy,
            "environment_kind",
            "docker",
        )

        if environment_kind == "fake":
            return FakeEnvironment()

        if environment_kind == "remote":
            contract = _build_remote_contract(
                plan,
                sample_payload,
                sandbox_policy,
                profiles=self._profiles,
                resources=resources,
            )
            return RemoteEnvironment(
                contract=contract,
                runtime_configs=runtime_configs,
                resources=resources,
                config=_build_remote_start_config(runtime_spec, contract, sample_payload),
            )

        return DockerEnvironment(
            runtime_configs=runtime_configs,
            resources=resources,
            config=_build_docker_start_config(runtime_spec, sample_payload),
        )


def _merge_dicts(base: Optional[Mapping[str, Any]], overrides: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base or {})
    if isinstance(overrides, Mapping):
        merged.update(overrides)
    return merged


def _build_runtime_configs(runtime_spec, resource_policy, sample: Mapping[str, Any]) -> dict[str, Any]:
    runtime_params = dict(getattr(runtime_spec, "params", {}) or {})
    nested_runtime_configs = runtime_params.pop("runtime_configs", None)
    runtime_configs = _merge_dicts(runtime_params, nested_runtime_configs)
    runtime_configs = _merge_dicts(runtime_configs, sample.get("runtime_configs"))
    explicit_env = runtime_configs.pop("env", None)
    env = _merge_dicts(getattr(resource_policy, "env", None), sample.get("env"))
    env = _merge_dicts(env, sample.get("runtime_env"))
    env = _merge_dicts(env, explicit_env)
    if env:
        runtime_configs["env"] = env
    return runtime_configs


def _build_resources(resource_policy, sample: Mapping[str, Any]) -> dict[str, Any]:
    resources = _merge_dicts(getattr(resource_policy, "params", None), sample.get("resources"))
    resources = _merge_dicts(resources, sample.get("resource_params"))
    return resources


def _build_remote_contract(
    plan,
    sample: Mapping[str, Any],
    sandbox_policy,
    *,
    profiles: Optional[Mapping[str, Mapping[str, Any]]],
    resources: Mapping[str, Any],
) -> RemoteSandboxContract:
    sample_remote_payload = sample.get("remote_sandbox") if isinstance(sample.get("remote_sandbox"), Mapping) else {}
    sample_sandbox_payload = sample.get("sandbox") if isinstance(sample.get("sandbox"), Mapping) else {}
    runtime_spec = getattr(plan, "runtime_spec", None)
    runtime_params = dict(getattr(runtime_spec, "params", {}) or {})
    runtime_layer = _merge_runtime_remote_layer(runtime_params)
    policy_layer = _build_policy_remote_layer(plan, sandbox_policy)
    profile_id = _resolve_sandbox_profile_id(plan, sandbox_policy, runtime_layer, sample_sandbox_payload, sample_remote_payload)
    merged = merge_sandbox_profile_layers(
        profiles,
        profile_id,
        {"mode": getattr(sandbox_policy, "remote_mode", None) or getattr(plan, "remote_mode", None) or runtime_layer.get("mode") or "attached"},
        runtime_layer,
        policy_layer,
        sample_sandbox_payload,
        sample_remote_payload,
    )

    auth = _build_auth_payload(merged)
    timeouts = _build_timeout_payload(merged)
    retries = _build_retry_payload(merged)
    contract_resources = _merge_dicts(resources, merged.get("resources"))
    params = _build_contract_params(merged)

    return RemoteSandboxContract(
        mode=str(merged.get("mode") or "attached"),
        sandbox_profile_id=profile_id,
        provider=_first_non_empty(
            merged.get("provider"),
            merged.get("runtime"),
            merged.get("backend"),
        ),
        sandbox_id=_first_non_empty(merged.get("sandbox_id"), merged.get("id")),
        image=_first_non_empty(merged.get("image"), merged.get("sandbox_image")),
        workspace_root=_first_non_empty(merged.get("workspace_root"), merged.get("cwd")),
        attach_target=_first_non_empty(merged.get("attach_target"), merged.get("workspace_root")),
        control_endpoint=_first_non_empty(merged.get("control_endpoint"), merged.get("control_url")),
        exec_endpoint=_first_non_empty(merged.get("exec_endpoint"), merged.get("exec_url")),
        file_endpoint=_first_non_empty(merged.get("file_endpoint"), merged.get("data_endpoint")),
        file_read_url=_first_non_empty(merged.get("file_read_url"), merged.get("read_file_url")),
        file_write_url=_first_non_empty(merged.get("file_write_url"), merged.get("write_file_url")),
        env_endpoint=_first_non_empty(merged.get("env_endpoint"), merged.get("environment_endpoint")),
        apis_endpoint=merged.get("apis_endpoint"),
        mcp_endpoint=merged.get("mcp_endpoint"),
        renew_supported=bool(merged.get("renew_supported")),
        auth=auth,
        timeouts=timeouts,
        retries=retries,
        resources=contract_resources,
        params=params,
    )
def _build_remote_start_config(runtime_spec, contract: RemoteSandboxContract, sample: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = _build_start_config(runtime_spec, sample)
    runtime_configs = dict(config.get("runtime_configs") or {})
    config["runtime_configs"] = runtime_configs
    if contract.control_endpoint:
        config.setdefault("control_endpoint", contract.control_endpoint)
    if contract.exec_endpoint:
        config.setdefault("exec_url", contract.exec_endpoint)
    if contract.file_endpoint:
        config.setdefault("data_endpoint", contract.file_endpoint)
    elif contract.exec_endpoint:
        config.setdefault("data_endpoint", _derive_base_endpoint(contract.exec_endpoint))
    if contract.file_read_url:
        config.setdefault("file_read_url", contract.file_read_url)
    if contract.file_write_url:
        config.setdefault("file_write_url", contract.file_write_url)
    if contract.env_endpoint:
        config.setdefault("env_endpoint", contract.env_endpoint)
    if contract.apis_endpoint:
        config.setdefault("apis_endpoint", contract.apis_endpoint)
    if contract.mcp_endpoint:
        config.setdefault("mcp_endpoint", contract.mcp_endpoint)
    if contract.attach_target:
        runtime_configs.setdefault("attach_target", contract.attach_target)
        config.setdefault("attach_target", contract.attach_target)
    if contract.workspace_root:
        runtime_configs.setdefault("workspace_root", contract.workspace_root)
        config.setdefault("workspace_root", contract.workspace_root)
    if contract.provider:
        config.setdefault("provider", contract.provider)
    if contract.sandbox_id:
        config.setdefault("sandbox_id", contract.sandbox_id)
    if contract.image:
        config.setdefault("image", contract.image)
    if contract.auth:
        _inject_auth_config(runtime_configs, contract.auth)
    if contract.timeouts:
        _inject_timeout_config(runtime_configs, contract.timeouts)
    if contract.retries:
        _inject_retry_config(runtime_configs, contract.retries)
    if contract.resources:
        config.setdefault("resources", dict(contract.resources))
    if contract.params:
        config.setdefault("params", deep_merge_dicts(config.get("params"), contract.params))
    runtime_configs.setdefault("remote_mode", contract.mode)
    return config


def _build_docker_start_config(runtime_spec, sample: Mapping[str, Any]) -> dict[str, Any]:
    config = _build_start_config(runtime_spec, sample)
    runtime_configs = dict(config.get("runtime_configs") or {})
    config["runtime_configs"] = runtime_configs
    return config


def _build_start_config(runtime_spec, sample: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = _extract_sandbox_payload(sample)
    runtime_params = dict(getattr(runtime_spec, "params", {}) or {})
    config.update({k: v for k, v in runtime_params.items() if k != "runtime_configs"})
    if "runtime_configs" in config and not isinstance(config["runtime_configs"], Mapping):
        config["runtime_configs"] = {}
    return config


def _extract_sandbox_payload(sample: Mapping[str, Any]) -> dict[str, Any]:
    for key in ("sandbox", "sandbox_config", "remote_sandbox"):
        candidate = sample.get(key)
        if isinstance(candidate, Mapping):
            return dict(candidate)
    return {}


def _as_mapping(sample: Any) -> Mapping[str, Any]:
    if isinstance(sample, Mapping):
        return sample
    return {}


def _derive_base_endpoint(url: str) -> str:
    from urllib.parse import urlsplit, urlunsplit

    split = urlsplit(url)
    path = split.path.rstrip("/")
    if not path:
        return url
    if "/" not in path:
        new_path = ""
    else:
        new_path = path.rsplit("/", 1)[0]
    return urlunsplit((split.scheme, split.netloc, new_path, split.query, split.fragment))


def _first_non_empty(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _resolve_sandbox_profile_id(
    plan,
    sandbox_policy,
    runtime_layer: Mapping[str, Any],
    sample_sandbox_payload: Mapping[str, Any],
    sample_remote_payload: Mapping[str, Any],
) -> Optional[str]:
    candidates = (
        sample_remote_payload.get("sandbox_profile_id"),
        sample_sandbox_payload.get("sandbox_profile_id"),
        getattr(sandbox_policy, "sandbox_profile_id", None),
        getattr(plan, "sandbox_profile_id", None),
        runtime_layer.get("sandbox_profile_id"),
    )
    return _first_non_empty(*candidates)


def _merge_runtime_remote_layer(runtime_params: Mapping[str, Any]) -> dict[str, Any]:
    params = dict(runtime_params or {})
    nested_runtime_configs = params.pop("runtime_configs", None)
    merged = deep_merge_dicts(params, nested_runtime_configs if isinstance(nested_runtime_configs, Mapping) else {})
    if "remote_mode" in merged and "mode" not in merged:
        merged["mode"] = merged["remote_mode"]
    if "headers" in params and "auth" not in merged:
        merged["auth"] = {"headers": dict(params.get("headers") or {})}
    return merged


def _build_policy_remote_layer(plan, sandbox_policy) -> dict[str, Any]:
    if sandbox_policy is None:
        sandbox_policy = getattr(getattr(plan, "runtime_spec", None), "sandbox_policy", None)
    params = dict(getattr(sandbox_policy, "params", {}) or {}) if sandbox_policy else {}
    payload = deep_merge_dicts({}, params)
    remote_mode = getattr(sandbox_policy, "remote_mode", None) if sandbox_policy else None
    if remote_mode:
        payload["mode"] = remote_mode
    sandbox_profile_id = getattr(sandbox_policy, "sandbox_profile_id", None) if sandbox_policy else None
    if sandbox_profile_id:
        payload["sandbox_profile_id"] = sandbox_profile_id
    return payload


def _build_auth_payload(merged: Mapping[str, Any]) -> dict[str, Any]:
    auth = deep_merge_dicts({}, merged.get("auth") if isinstance(merged.get("auth"), Mapping) else {})
    headers = merged.get("headers")
    if isinstance(headers, Mapping):
        auth["headers"] = _merge_dicts(auth.get("headers"), headers)
    if merged.get("auth_type") and "type" not in auth:
        auth["type"] = merged.get("auth_type")
    if merged.get("auth_token") and "token" not in auth:
        auth["token"] = merged.get("auth_token")
    if merged.get("api_key_header") and "api_key_header" not in auth:
        auth["api_key_header"] = merged.get("api_key_header")
    return auth


def _build_timeout_payload(merged: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for source_key, target_key in (
        ("timeout_s", "exec_timeout_s"),
        ("file_timeout_s", "file_timeout_s"),
        ("startup_timeout_s", "startup_timeout_s"),
    ):
        value = merged.get(source_key)
        if value is not None:
            payload[target_key] = value
    return payload


def _build_retry_payload(merged: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    if merged.get("max_retries") is not None:
        payload["max_retries"] = merged.get("max_retries")
    if merged.get("retry_backoff_factor") is not None:
        payload["retry_backoff_factor"] = merged.get("retry_backoff_factor")
    return payload


def _build_contract_params(merged: Mapping[str, Any]) -> dict[str, Any]:
    fixed_keys = {
        "mode",
        "remote_mode",
        "sandbox_profile_id",
        "provider",
        "runtime",
        "backend",
        "sandbox_id",
        "id",
        "image",
        "sandbox_image",
        "workspace_root",
        "cwd",
        "attach_target",
        "control_endpoint",
        "control_url",
        "exec_endpoint",
        "exec_url",
        "file_endpoint",
        "data_endpoint",
        "file_read_url",
        "read_file_url",
        "file_write_url",
        "write_file_url",
        "env_endpoint",
        "environment_endpoint",
        "apis_endpoint",
        "mcp_endpoint",
        "renew_supported",
        "auth",
        "headers",
        "auth_type",
        "auth_token",
        "api_key_header",
        "timeout_s",
        "file_timeout_s",
        "startup_timeout_s",
        "max_retries",
        "retry_backoff_factor",
        "resources",
        "params",
    }
    params = deep_merge_dicts({}, merged.get("params") if isinstance(merged.get("params"), Mapping) else {})
    extras = {str(key): value for key, value in merged.items() if key not in fixed_keys}
    return deep_merge_dicts(params, extras)


def _inject_auth_config(runtime_configs: dict[str, Any], auth: Mapping[str, Any]) -> None:
    headers = auth.get("headers")
    if isinstance(headers, Mapping):
        runtime_configs.setdefault("headers", {})
        runtime_configs["headers"] = _merge_dicts(runtime_configs.get("headers"), headers)
    if auth.get("type") is not None:
        runtime_configs.setdefault("auth_type", auth.get("type"))
    if auth.get("token") is not None:
        runtime_configs.setdefault("auth_token", auth.get("token"))
    if auth.get("api_key_header") is not None:
        runtime_configs.setdefault("api_key_header", auth.get("api_key_header"))


def _inject_timeout_config(runtime_configs: dict[str, Any], timeouts: Mapping[str, Any]) -> None:
    if timeouts.get("exec_timeout_s") is not None:
        runtime_configs.setdefault("timeout_s", timeouts.get("exec_timeout_s"))
    if timeouts.get("file_timeout_s") is not None:
        runtime_configs.setdefault("file_timeout_s", timeouts.get("file_timeout_s"))
    if timeouts.get("startup_timeout_s") is not None:
        runtime_configs.setdefault("startup_timeout_s", timeouts.get("startup_timeout_s"))


def _inject_retry_config(runtime_configs: dict[str, Any], retries: Mapping[str, Any]) -> None:
    if retries.get("max_retries") is not None:
        runtime_configs.setdefault("max_retries", retries.get("max_retries"))
    if retries.get("retry_backoff_factor") is not None:
        runtime_configs.setdefault("retry_backoff_factor", retries.get("retry_backoff_factor"))
