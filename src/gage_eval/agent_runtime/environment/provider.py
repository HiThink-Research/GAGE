"""Environment construction helpers."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from gage_eval.agent_runtime.environment.base import AgentEnvironment
from gage_eval.agent_runtime.environment.docker_environment import DockerEnvironment
from gage_eval.agent_runtime.environment.fake import FakeEnvironment
from gage_eval.agent_runtime.environment.remote_environment import RemoteEnvironment
from gage_eval.agent_runtime.resources.remote_sandbox import RemoteSandboxContract
from gage_eval.agent_runtime.resources.sandbox_policy import resolve_sandbox_policy


class EnvironmentProvider:
    """Phase 1 main-path factory."""

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
            contract = _build_remote_contract(plan, sample_payload, sandbox_policy)
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


def _build_remote_contract(plan, sample: Mapping[str, Any], sandbox_policy) -> RemoteSandboxContract:
    remote_payload = _extract_remote_payload(sample)
    runtime_spec = getattr(plan, "runtime_spec", None)
    runtime_params = dict(getattr(runtime_spec, "params", {}) or {})
    policy_params = dict(getattr(sandbox_policy, "params", {}) or {}) if sandbox_policy else {}

    mode = (
        remote_payload.get("mode")
        or getattr(sandbox_policy, "remote_mode", None)
        or runtime_params.get("remote_mode")
        or "attached"
    )
    control_endpoint = _first_non_empty(
        remote_payload.get("control_endpoint"),
        runtime_params.get("control_endpoint"),
        policy_params.get("control_endpoint"),
    )
    exec_endpoint = _first_non_empty(
        remote_payload.get("exec_endpoint"),
        remote_payload.get("exec_url"),
        runtime_params.get("exec_endpoint"),
        runtime_params.get("exec_url"),
        policy_params.get("exec_endpoint"),
        policy_params.get("exec_url"),
    )
    file_endpoint = _first_non_empty(
        remote_payload.get("file_endpoint"),
        runtime_params.get("file_endpoint"),
        policy_params.get("file_endpoint"),
    )
    attach_target = _first_non_empty(
        remote_payload.get("attach_target"),
        runtime_params.get("attach_target"),
        policy_params.get("attach_target"),
    )
    renew_supported = bool(
        remote_payload.get("renew_supported")
        or runtime_params.get("renew_supported")
        or policy_params.get("renew_supported")
    )
    params = _merge_dicts(runtime_params, remote_payload.get("params"))
    params.update(policy_params)

    return RemoteSandboxContract(
        mode=mode,
        control_endpoint=control_endpoint,
        exec_endpoint=exec_endpoint,
        file_endpoint=file_endpoint,
        attach_target=attach_target,
        renew_supported=renew_supported,
        params=params,
    )


def _extract_remote_payload(sample: Mapping[str, Any]) -> Mapping[str, Any]:
    for key in ("remote_sandbox", "sandbox", "sandbox_config"):
        candidate = sample.get(key)
        if isinstance(candidate, Mapping):
            return candidate
    return {}


def _build_remote_start_config(runtime_spec, contract: RemoteSandboxContract, sample: Mapping[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = _build_start_config(runtime_spec, sample)
    runtime_configs = dict(config.get("runtime_configs") or {})
    config["runtime_configs"] = runtime_configs
    if contract.mode == "managed":
        if contract.control_endpoint:
            config.setdefault("control_endpoint", contract.control_endpoint)
        if contract.exec_endpoint:
            config.setdefault("exec_url", contract.exec_endpoint)
        if contract.file_endpoint:
            config.setdefault("data_endpoint", contract.file_endpoint)
    else:
        if contract.exec_endpoint:
            config.setdefault("exec_url", contract.exec_endpoint)
            config.setdefault("data_endpoint", _derive_base_endpoint(contract.exec_endpoint))
        if contract.file_endpoint:
            config.setdefault("data_endpoint", contract.file_endpoint)
    if contract.attach_target:
        runtime_configs.setdefault("attach_target", contract.attach_target)
        config.setdefault("attach_target", contract.attach_target)
    if contract.params:
        config.setdefault("params", dict(contract.params))
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
