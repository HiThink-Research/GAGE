"""Agent Eval Kit v2 configuration schema and loader."""

from __future__ import annotations

import os
import re
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from gage_eval.agent_runtime.clients.contracts import (
    AcpClientSchedulerConfig,
    InstalledClientSchedulerConfig,
)
from gage_eval.config.loader_cli import CLIIntent


class AgentKitV2ValidationError(ValueError):
    """Raised when an Agent Eval Kit v2 config is structurally invalid."""


@dataclass(frozen=True)
class AgentkitV2RuntimeBinding:
    """Runtime binding produced from one AgentKit v2 dut_agent entry."""

    dut_id: str
    agent_id: str
    benchmark_id: str
    env_id: str
    scheduler_type: str
    backend_id: str | None
    environment_provider: str
    executor_ref: Any


@dataclass(frozen=True)
class AgentkitV2RuntimeBindingSpec:
    """Pure runtime binding spec lowered from one AgentKit v2 dut_agent entry."""

    dut_id: str
    agent_id: str
    benchmark_id: str
    kit_id: str
    env_id: str
    scheduler_type: str
    scheduler_config: dict[str, Any]
    backend_id: str | None
    environment_provider: str
    environment_profile: dict[str, Any]
    provider_config: dict[str, Any]
    startup_env: dict[str, Any]
    resources: dict[str, Any]
    lifecycle: str
    agent_tooling: dict[str, Any]


_ENV_REF_PATTERN = re.compile(r"^\$\{([^}:]+)(?:(:-|:\?)(.*))?\}$")
_SECRET_KEYNAME_PATTERN = re.compile(
    r"(?i)(?:^|[_\-\s])("
    r"api[_-]?key|api[_-]?secret|api[_-]?token|"
    r"client[_-]?secret|access[_-]?token|bearer[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|auth[_-]?token|jwt[_-]?token|"
    r"csrf[_-]?token|session[_-]?token|secret|token|password|"
    r"passphrase|credential|credentials|private[_-]?key|signing[_-]?key|"
    r"authorization|auth"
    r")(?:$|[_\-\s])"
)
_USAGE_KEYNAME_ALLOWLIST = {
    "agent_total_tokens",
    "cached_tokens",
    "completion_tokens",
    "completion_tokens_details",
    "input_tokens",
    "input_tokens_details",
    "output_tokens",
    "output_tokens_details",
    "prompt_tokens",
    "prompt_tokens_details",
    "reasoning_tokens",
    "total_tokens",
    "user_total_tokens",
}
_LEGACY_TOP_LEVEL_KEYS = {
    "schema_version",
    "runtime_version",
    "agent_backends",
    "agent_backend_id",
    "benchmark_configs",
    "sandbox_profile_id",
    "kit",
    "scheduler",
    "environment",
}


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class BackendSpec(_StrictModel):
    backend_id: str
    type: str
    config: dict[str, Any] = Field(default_factory=dict)


class SchedulerSpec(_StrictModel):
    type: Literal["framework_loop", "installed_client", "acp_client"] = "framework_loop"
    backend_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_scheduler_config(self) -> "SchedulerSpec":
        if self.type == "installed_client":
            self.config = InstalledClientSchedulerConfig.model_validate(self.config).model_dump(
                mode="python",
                exclude_none=True,
            )
        elif self.type == "acp_client":
            self.config = AcpClientSchedulerConfig.model_validate(self.config).model_dump(
                mode="python",
                exclude_none=True,
            )
        return self


class AgentToolingSpec(_StrictModel):
    skill_ids: list[str] = Field(default_factory=list)
    mcp_servers: list[str] = Field(default_factory=list)
    skill_manifests: dict[str, dict[str, Any]] = Field(default_factory=dict)


class AgentSpec(_StrictModel):
    agent_id: str
    scheduler: SchedulerSpec
    config: dict[str, Any] = Field(default_factory=dict)
    tooling: AgentToolingSpec = Field(default_factory=AgentToolingSpec)


class BenchmarkSpec(_StrictModel):
    benchmark_id: str
    kit_id: str
    config: dict[str, Any] = Field(default_factory=dict)


class EnvironmentSpec(_StrictModel):
    env_id: str
    provider: Literal["local_process", "docker", "e2b"]
    profile_id: str
    profile: dict[str, Any]
    lifecycle: str = "per_sample"
    provider_config: dict[str, Any] = Field(default_factory=dict)
    startup_env: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _reject_legacy_environment_fields(self) -> "EnvironmentSpec":
        _fail_on_denied_mapping_keys(
            self.profile,
            denied={
                "runtime": "please remove profile.runtime and use provider",
                "runtime_configs": "please rename profile.runtime_configs to provider_config",
                "sandbox_id": "please use profile_id",
                "sandbox_runtime": "please use provider",
                "sandbox_lifecycle": "please use lifecycle",
                "provider_config": "please use environments[].provider_config, not profile.provider_config",
            },
            prefix=f"environments.{self.env_id}.profile",
        )
        _fail_on_denied_mapping_keys(
            self.provider_config,
            denied={"runtime_configs": "please flatten provider_config"},
            prefix=f"environments.{self.env_id}.provider_config",
        )
        _fail_on_denied_mapping_keys(
            self.resources,
            denied={"memory": "please use memory_gb as a numeric value"},
            prefix=f"environments.{self.env_id}.resources",
        )
        return self


class TrialPolicy(_StrictModel):
    trials: int = Field(default=1, ge=1)


class DutAgentSpec(_StrictModel):
    dut_id: str
    agent_id: str
    env_id: str
    benchmark_id: str
    trial_policy: TrialPolicy = Field(default_factory=TrialPolicy)


class AgentkitV2ConfigModel(_StrictModel):
    kind: Literal["AgentEvalConfig"]
    metadata: dict[str, Any] = Field(default_factory=dict)
    backends: list[BackendSpec]
    agents: list[AgentSpec]
    benchmarks: list[BenchmarkSpec]
    environments: list[EnvironmentSpec]
    dut_agents: list[DutAgentSpec]
    trial_policy: TrialPolicy = Field(default_factory=TrialPolicy)

    @model_validator(mode="after")
    def _validate_references(self) -> "AgentkitV2ConfigModel":
        _fail_on_duplicate_ids("backends", "backend_id", [backend.backend_id for backend in self.backends])
        _fail_on_duplicate_ids("agents", "agent_id", [agent.agent_id for agent in self.agents])
        _fail_on_duplicate_ids("benchmarks", "benchmark_id", [benchmark.benchmark_id for benchmark in self.benchmarks])
        _fail_on_duplicate_ids("environments", "env_id", [env.env_id for env in self.environments])
        _fail_on_duplicate_ids("dut_agents", "dut_id", [dut.dut_id for dut in self.dut_agents])

        backend_ids = {backend.backend_id for backend in self.backends}
        agent_ids = {agent.agent_id for agent in self.agents}
        env_ids = {env.env_id for env in self.environments}
        benchmark_ids = {benchmark.benchmark_id for benchmark in self.benchmarks}

        for agent in self.agents:
            backend_id = agent.scheduler.backend_id
            if agent.scheduler.type == "framework_loop" and not backend_id:
                raise ValueError(
                    f"config.reference.required agents.{agent.agent_id}.scheduler.backend_id.required"
                )
            if backend_id is not None and backend_id not in backend_ids:
                raise ValueError(
                    f"config.reference.missing agents.{agent.agent_id}.scheduler.backend_id={backend_id}"
                )

        for dut in self.dut_agents:
            if dut.agent_id not in agent_ids:
                raise ValueError(f"config.reference.missing dut_agents.{dut.dut_id}.agent_id={dut.agent_id}")
            if dut.env_id not in env_ids:
                raise ValueError(f"config.reference.missing dut_agents.{dut.dut_id}.env_id={dut.env_id}")
            if dut.benchmark_id not in benchmark_ids:
                raise ValueError(
                    f"config.reference.missing dut_agents.{dut.dut_id}.benchmark_id={dut.benchmark_id}"
                )

        for env in self.environments:
            if env.lifecycle == "per_task":
                raise ValueError(f"config.environment.lifecycle.per_task environments.{env.env_id}")
            if env.lifecycle != "per_sample":
                raise ValueError(f"config.environment.lifecycle.unsupported environments.{env.env_id}")
        return self


def load_agentkit_v2_config_payload(
    path: Path,
    *,
    cli_intent: CLIIntent | None = None,
) -> dict[str, Any]:
    """Load and materialize an Agent Eval Kit v2 YAML config."""

    if not path.exists():
        raise FileNotFoundError(f"Config file '{path}' not found")
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise AgentKitV2ValidationError(f"Config '{path}' must be a mapping at the top level")
    return materialize_agentkit_v2_config_payload(payload, path, cli_intent=cli_intent)


def materialize_agentkit_v2_config_payload(
    payload: dict[str, Any],
    source_path: Path | None,
    *,
    cli_intent: CLIIntent | None = None,
) -> dict[str, Any]:
    """Expand defaults, validate references, apply CLI intent, and redact effective config."""

    _fail_on_legacy_keys(payload)
    expanded = _expand_env_references(deepcopy(payload), redact=False)
    _apply_agentkit_v2_cli_overrides(expanded, cli_intent or CLIIntent())
    try:
        model = AgentkitV2ConfigModel.model_validate(expanded)
        kit_configs = _validate_benchmark_kit_configs(model, payload)
    except ValidationError as exc:
        raise AgentKitV2ValidationError(_sanitize_env_reference_values(str(exc), payload)) from exc
    except ValueError as exc:
        raise AgentKitV2ValidationError(_sanitize_env_reference_values(str(exc), payload)) from exc

    materialized = model.model_dump(mode="python")
    _apply_validated_benchmark_configs(materialized, kit_configs)
    redacted = _expand_env_references(deepcopy(payload), redact=True)
    _apply_agentkit_v2_cli_overrides(redacted, cli_intent or CLIIntent())
    materialized = _overlay_redacted_references(materialized, redacted)
    materialized = _redact_sensitive_keynames(materialized)
    trial_policy = dict(materialized["trial_policy"])
    raw_duts = [dut for dut in expanded.get("dut_agents") or [] if isinstance(dut, dict)]
    for index, dut in enumerate(materialized["dut_agents"]):
        if index >= len(raw_duts) or "trial_policy" not in raw_duts[index]:
            dut["trial_policy"] = deepcopy(trial_policy)
    materialized["effective_config"] = deepcopy(materialized)
    return materialized


def materialize_agentkit_v2_runtime_config_payload(
    payload: dict[str, Any],
    source_path: Path | None,
    *,
    cli_intent: CLIIntent | None = None,
) -> dict[str, Any]:
    """Materialize AgentKit v2 config for private runtime use with resolved secrets."""

    del source_path
    _fail_on_legacy_keys(payload)
    expanded = _expand_env_references(deepcopy(payload), redact=False)
    _apply_agentkit_v2_cli_overrides(expanded, cli_intent or CLIIntent())
    try:
        model = AgentkitV2ConfigModel.model_validate(expanded)
        kit_configs = _validate_benchmark_kit_configs(model, payload)
    except ValidationError as exc:
        raise AgentKitV2ValidationError(_sanitize_env_reference_values(str(exc), payload)) from exc
    except ValueError as exc:
        raise AgentKitV2ValidationError(_sanitize_env_reference_values(str(exc), payload)) from exc

    runtime_config = model.model_dump(mode="python")
    _apply_validated_benchmark_configs(runtime_config, kit_configs)
    trial_policy = dict(runtime_config["trial_policy"])
    raw_duts = [dut for dut in expanded.get("dut_agents") or [] if isinstance(dut, dict)]
    for index, dut in enumerate(runtime_config["dut_agents"]):
        if index >= len(raw_duts) or "trial_policy" not in raw_duts[index]:
            dut["trial_policy"] = deepcopy(trial_policy)
    return runtime_config


def build_agentkit_v2_runtime_bindings(
    materialized_config: dict[str, Any],
    *,
    backends: dict[str, Any],
    runtime_config: dict[str, Any] | None = None,
    installed_client_overrides: dict[str, Any] | None = None,
    mcp_clients: dict[str, Any] | None = None,
    sandbox_manager: Any | None = None,
) -> dict[str, AgentkitV2RuntimeBinding]:
    """Build runtime executors from materialized AgentKit v2 scheduler bindings.

    The helper is intentionally explicit: v2 framework-loop agents resolve
    ``agents[].scheduler.backend_id`` to top-level static model backends and
    pass those materialized objects into the runtime resolver.
    """

    if runtime_config is None:
        raise AgentKitV2ValidationError("config.runtime_config.required")

    from gage_eval.agent_runtime.resolver import (
        build_compiled_runtime_executor,
        compile_agent_runtime_plan,
    )

    bindings: dict[str, AgentkitV2RuntimeBinding] = {}

    for spec in resolve_agentkit_v2_runtime_binding_specs(
        materialized_config,
        runtime_config=runtime_config,
    ).values():
        static_model_backend = None
        if spec.scheduler_type == "framework_loop":
            if spec.backend_id not in backends:
                raise KeyError(
                    f"Backend '{spec.backend_id}' referenced by agent '{spec.agent_id}' is not materialized"
                )
            static_model_backend = backends[spec.backend_id]
        elif spec.backend_id is not None and spec.backend_id not in backends:
            raise KeyError(
                f"Backend '{spec.backend_id}' referenced by agent '{spec.agent_id}' is not materialized"
            )

        compiled_plan = compile_agent_runtime_plan(
            agent_runtime_id=_agent_runtime_id_for(
                kit_id=spec.kit_id,
                scheduler_type=spec.scheduler_type,
            ),
            environment_profile=spec.environment_profile,
            provider_config=spec.provider_config,
            resources=spec.resources,
            startup_env=spec.startup_env,
            lifecycle=spec.lifecycle,
        )
        compiled_plan = _with_environment_provider(compiled_plan, spec.environment_provider)
        compiled_plan = replace(
            compiled_plan,
            scheduler_config=dict(spec.scheduler_config or {}),
            agent_config={
                **dict(compiled_plan.agent_config or {}),
                "tooling": dict(spec.agent_tooling or {}),
            },
        )
        executor_ref = build_compiled_runtime_executor(
            compiled_plan=compiled_plan,
            agent_backend=None,
            static_model_backend=static_model_backend,
            installed_client_override=(installed_client_overrides or {}).get(spec.agent_id),
            mcp_clients=mcp_clients,
            sandbox_manager=sandbox_manager,
        )
        bindings[spec.dut_id] = AgentkitV2RuntimeBinding(
            dut_id=spec.dut_id,
            agent_id=spec.agent_id,
            benchmark_id=spec.benchmark_id,
            env_id=spec.env_id,
            scheduler_type=spec.scheduler_type,
            backend_id=spec.backend_id,
            environment_provider=spec.environment_provider,
            executor_ref=executor_ref,
        )
    return bindings


def resolve_agentkit_v2_runtime_binding_specs(
    materialized_config: dict[str, Any],
    *,
    runtime_config: dict[str, Any] | None = None,
) -> dict[str, AgentkitV2RuntimeBindingSpec]:
    """Lower AgentKit v2 dut/agent/backend/env references without side effects."""

    if runtime_config is None:
        raise AgentKitV2ValidationError("config.runtime_config.required")

    runtime_source = runtime_config
    agents = _index_by_id(runtime_source.get("agents") or [], "agent_id")
    benchmarks = _index_by_id(runtime_source.get("benchmarks") or [], "benchmark_id")
    environments = _index_by_id(runtime_source.get("environments") or [], "env_id")
    backend_ids = {
        str(spec["backend_id"])
        for spec in runtime_source.get("backends") or []
        if isinstance(spec, dict) and spec.get("backend_id")
    }
    specs: dict[str, AgentkitV2RuntimeBindingSpec] = {}

    for dut in runtime_source.get("dut_agents") or []:
        if not isinstance(dut, dict):
            continue
        dut_id = str(dut.get("dut_id") or "")
        agent = agents[str(dut.get("agent_id") or "")]
        benchmark = benchmarks[str(dut.get("benchmark_id") or "")]
        environment = environments[str(dut.get("env_id") or "")]
        scheduler = agent.get("scheduler") or {}
        scheduler_type = str(scheduler.get("type") or "framework_loop")
        backend_id = scheduler.get("backend_id")
        if scheduler_type == "framework_loop":
            if not isinstance(backend_id, str) or not backend_id:
                raise AgentKitV2ValidationError(
                    f"config.reference.required agents.{agent['agent_id']}.scheduler.backend_id.required"
                )
            if backend_id not in backend_ids:
                raise KeyError(
                    f"Backend '{backend_id}' referenced by agent '{agent['agent_id']}' is not declared"
                )
        elif backend_id is not None and backend_id not in backend_ids:
            raise KeyError(
                f"Backend '{backend_id}' referenced by agent '{agent['agent_id']}' is not declared"
            )
        provider = str(environment.get("provider") or "")
        if provider not in {"local_process", "docker", "e2b"}:
            raise AgentKitV2ValidationError(
                f"config.environment.provider.unsupported environments.{environment.get('env_id')}"
            )
        provider_config = _provider_config_from_environment(environment)
        provider_config = _provider_config_with_benchmark_runtime_overrides(
            provider_config,
            kit_id=str(benchmark.get("kit_id") or benchmark.get("benchmark_id") or ""),
            benchmark_config=dict(benchmark.get("config") or {}),
        )
        specs[dut_id] = AgentkitV2RuntimeBindingSpec(
            dut_id=dut_id,
            agent_id=str(agent.get("agent_id") or ""),
            benchmark_id=str(benchmark.get("benchmark_id") or ""),
            kit_id=str(benchmark.get("kit_id") or benchmark.get("benchmark_id") or ""),
            env_id=str(environment.get("env_id") or ""),
            scheduler_type=scheduler_type,
            scheduler_config=dict(scheduler.get("config") or {}),
            backend_id=backend_id if isinstance(backend_id, str) else None,
            environment_provider=provider,
            environment_profile=_environment_profile_from_environment(environment),
            provider_config=provider_config,
            startup_env=dict(environment.get("startup_env") or {}),
            resources=dict(environment.get("resources") or {}),
            lifecycle=str(environment.get("lifecycle") or "per_sample"),
            agent_tooling=dict(agent.get("tooling") or {}),
        )
    return specs


def lower_agentkit_v2_pipeline_payload(
    payload: dict[str, Any],
    *,
    source_path: Path | None = None,
    cli_intent: CLIIntent | None = None,
) -> dict[str, Any]:
    """Generate PipelineConfig role_adapters from AgentKit v2 sections.

    This keeps manual executable configs aligned with the v2 public shape while
    the current PipelineConfig runner still consumes role_adapters[].
    """

    v2_payload = {
        "kind": "AgentEvalConfig",
        "metadata": deepcopy(payload.get("metadata") or {}),
        "backends": deepcopy(payload.get("backends") or []),
        "agents": deepcopy(payload.get("agents") or []),
        "benchmarks": deepcopy(payload.get("benchmarks") or []),
        "environments": deepcopy(payload.get("environments") or []),
        "dut_agents": deepcopy(payload.get("dut_agents") or []),
    }
    if "trial_policy" in payload:
        v2_payload["trial_policy"] = deepcopy(payload["trial_policy"])

    runtime_config = materialize_agentkit_v2_runtime_config_payload(
        v2_payload,
        source_path,
        cli_intent=cli_intent,
    )
    specs = resolve_agentkit_v2_runtime_binding_specs(
        runtime_config,
        runtime_config=runtime_config,
    )
    agents = _index_by_id(runtime_config.get("agents") or [], "agent_id")
    benchmarks = _index_by_id(runtime_config.get("benchmarks") or [], "benchmark_id")

    lowered = deepcopy(payload)
    lowered["agents"] = deepcopy(runtime_config.get("agents") or [])
    lowered["benchmarks"] = deepcopy(runtime_config.get("benchmarks") or [])
    lowered["environments"] = deepcopy(runtime_config.get("environments") or [])
    lowered["dut_agents"] = deepcopy(runtime_config.get("dut_agents") or [])
    lowered["role_adapters"] = [
        _role_adapter_from_v2_binding(
            spec,
            agent=agents[spec.agent_id],
            benchmark=benchmarks[spec.benchmark_id],
        )
        for spec in specs.values()
    ]
    return lowered


def _role_adapter_from_v2_binding(
    spec: AgentkitV2RuntimeBindingSpec,
    *,
    agent: dict[str, Any],
    benchmark: dict[str, Any],
) -> dict[str, Any]:
    scheduler_config = dict(spec.scheduler_config or {})
    agent_config = dict(agent.get("config") or {})
    role_adapter: dict[str, Any] = {
        "adapter_id": spec.dut_id,
        "role_type": "dut_agent",
        "agent_runtime_id": _agent_runtime_id_for(
            kit_id=spec.kit_id,
            scheduler_type=spec.scheduler_type,
        ),
    }
    if spec.backend_id:
        role_adapter["backend_id"] = spec.backend_id

    prompt_id = agent_config.get("prompt_id")
    if isinstance(prompt_id, str) and prompt_id:
        role_adapter["prompt_id"] = prompt_id
    prompt_params = agent_config.get("prompt_params") or agent_config.get("prompt_args")
    if isinstance(prompt_params, dict) and prompt_params:
        role_adapter["prompt_params"] = dict(prompt_params)

    params: dict[str, Any] = {}
    params["environment_profile"] = deepcopy(spec.environment_profile)
    params["provider_config"] = deepcopy(spec.provider_config)
    params["resources"] = deepcopy(spec.resources)
    params["startup_env"] = deepcopy(spec.startup_env)
    params["lifecycle"] = spec.lifecycle
    if "max_turns" in scheduler_config:
        params["max_turns"] = scheduler_config["max_turns"]
    if "cost_limit_usd" in scheduler_config:
        params["cost_limit_usd"] = scheduler_config["cost_limit_usd"]
    if params:
        role_adapter["params"] = params
    return role_adapter


def _provider_config_with_benchmark_runtime_overrides(
    provider_config: dict[str, Any],
    *,
    kit_id: str,
    benchmark_config: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(provider_config)
    if kit_id == "tau2" and isinstance(benchmark_config.get("user_simulator"), dict):
        config.setdefault("user_simulator", deepcopy(benchmark_config["user_simulator"]))
    return config


def _fail_on_legacy_keys(payload: dict[str, Any]) -> None:
    for key in _LEGACY_TOP_LEVEL_KEYS:
        if key in payload:
            raise AgentKitV2ValidationError(f"config.legacy_key.{key}")


def _fail_on_duplicate_ids(section: str, id_field: str, ids: list[str]) -> None:
    seen: set[str] = set()
    for item_id in ids:
        if item_id in seen:
            raise ValueError(f"config.duplicate_id.{section}.{id_field}={item_id}")
        seen.add(item_id)


def _index_by_id(items: list[Any], id_field: str) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    for item in items:
        if isinstance(item, dict) and item.get(id_field):
            indexed[str(item[id_field])] = item
    return indexed


def _agent_runtime_id_for(*, kit_id: str, scheduler_type: str) -> str:
    return f"{kit_id}_{scheduler_type}"


def _with_environment_provider(compiled_plan: Any, environment_provider: str) -> Any:
    resource_plan = dict(compiled_plan.resource_plan or {})
    resource_plan["resource_kind"] = environment_provider
    environment_profile = dict(compiled_plan.environment_profile or {})
    environment_profile["provider"] = environment_provider
    resource_plan["environment_profile"] = environment_profile
    verifier_profile_id = compiled_plan.verifier_environment_profile_id
    if compiled_plan.verifier_environment_policy == "fresh_from_profile":
        profile_by_provider = getattr(
            getattr(compiled_plan, "kit_entry", None),
            "default_environment_profile_by_provider",
            {},
        )
        if isinstance(profile_by_provider, dict):
            verifier_profile_id = profile_by_provider.get(environment_provider) or verifier_profile_id
    return replace(
        compiled_plan,
        environment_provider=environment_provider,
        environment_profile=environment_profile,
        resource_plan=resource_plan,
        verifier_environment_profile_id=verifier_profile_id,
    )


def _environment_profile_from_environment(environment: dict[str, Any]) -> dict[str, Any]:
    profile = dict(environment.get("profile") or {})
    provider = environment.get("provider")
    if isinstance(provider, str) and provider:
        profile.setdefault("provider", provider)
    profile_id = environment.get("profile_id")
    if isinstance(profile_id, str) and profile_id:
        profile.setdefault("profile_id", profile_id)
    provider_config = environment.get("provider_config")
    if isinstance(provider_config, dict):
        profile["config"] = _deep_merge(dict(profile.get("config") or {}), dict(provider_config))
    resources = environment.get("resources")
    if isinstance(resources, dict) and resources:
        profile.setdefault("resources", dict(resources))
    startup_env = environment.get("startup_env")
    if isinstance(startup_env, dict) and startup_env:
        profile.setdefault("startup_env", dict(startup_env))
    return profile


def _provider_config_from_environment(environment: dict[str, Any]) -> dict[str, Any]:
    profile = dict(environment.get("profile") or {})
    config: dict[str, Any] = {}
    for key in ("config", "provider_config"):
        value = profile.get(key)
        if isinstance(value, dict):
            config = _deep_merge(config, value)
    provider_config = environment.get("provider_config")
    if isinstance(provider_config, dict):
        config = _deep_merge(config, provider_config)
    return config


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


def _fail_on_denied_mapping_keys(
    value: dict[str, Any],
    *,
    denied: dict[str, str],
    prefix: str,
) -> None:
    if not isinstance(value, dict):
        return
    for key, hint in denied.items():
        if key in value:
            raise ValueError(f"config.legacy_key.{prefix}.{key}: {hint}")


def _overlay_redacted_references(target: Any, redacted_source: Any) -> Any:
    if isinstance(redacted_source, str) and redacted_source.startswith("<redacted:reference:"):
        return redacted_source
    if isinstance(target, dict) and isinstance(redacted_source, dict):
        merged = dict(target)
        for key, value in redacted_source.items():
            if key in merged:
                merged[key] = _overlay_redacted_references(merged[key], value)
        return merged
    if isinstance(target, list) and isinstance(redacted_source, list):
        merged = list(target)
        for index, value in enumerate(redacted_source):
            if index < len(merged):
                merged[index] = _overlay_redacted_references(merged[index], value)
        return merged
    return target


def _redact_sensitive_keynames(value: Any, *, key_name: str | None = None) -> Any:
    if key_name is not None and _is_secret_keyname(str(key_name)):
        if value in (None, ""):
            return value
        if isinstance(value, str):
            if value.startswith("<redacted") or value.startswith("${") or not value.strip():
                return value
        return f"<redacted:keyname:{key_name}>"
    if isinstance(value, dict):
        return {key: _redact_sensitive_keynames(child, key_name=str(key)) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_sensitive_keynames(child) for child in value]
    return value


def _is_secret_keyname(key_name: str) -> bool:
    if str(key_name or "").lower() in _USAGE_KEYNAME_ALLOWLIST:
        return False
    return bool(_SECRET_KEYNAME_PATTERN.search(str(key_name or "")))


def _validate_benchmark_kit_configs(
    model: AgentkitV2ConfigModel,
    original_payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    from gage_eval.agent_eval_kits import load_benchmark_kit

    validated: dict[str, dict[str, Any]] = {}
    for benchmark in model.benchmarks:
        try:
            entry = load_benchmark_kit(benchmark.kit_id)
        except KeyError as exc:
            raise ValueError(f"config.reference.missing benchmarks.{benchmark.benchmark_id}.kit_id={benchmark.kit_id}") from exc
        try:
            kit_config = entry.config_schema.model_validate(benchmark.config)
        except ValidationError as exc:
            code = "unknown_field" if any(error.get("type") == "extra_forbidden" for error in exc.errors()) else "validation_failed"
            message = f"config.kit_schema.{code} benchmarks.{benchmark.benchmark_id}.config {exc}"
            raise ValueError(_sanitize_env_reference_values(message, original_payload)) from exc
        validated[benchmark.benchmark_id] = kit_config.model_dump(mode="python", exclude_none=True)
    return validated


def _apply_validated_benchmark_configs(payload: dict[str, Any], kit_configs: dict[str, dict[str, Any]]) -> None:
    for benchmark in payload.get("benchmarks") or []:
        if isinstance(benchmark, dict) and benchmark.get("benchmark_id") in kit_configs:
            benchmark["config"] = deepcopy(kit_configs[str(benchmark["benchmark_id"])])


def _sanitize_env_reference_values(message: str, payload: Any) -> str:
    sanitized = message
    for resolved, redacted in _env_reference_redactions(payload):
        sanitized = _replace_outside_redacted_markers(sanitized, resolved, redacted)
    return sanitized


def _env_reference_redactions(value: Any) -> list[tuple[str, str]]:
    redactions: dict[str, str] = {}

    def visit(item: Any) -> None:
        if isinstance(item, dict):
            for child in item.values():
                visit(child)
            return
        if isinstance(item, list):
            for child in item:
                visit(child)
            return
        if not isinstance(item, str):
            return

        match = _ENV_REF_PATTERN.match(item)
        if not match:
            return
        reference, operator, operand = match.group(1), match.group(2), match.group(3)
        env_name = reference[4:] if reference.startswith("ENV.") else reference
        resolved = os.getenv(env_name, operand if operator == ":-" and operand is not None else "")
        if isinstance(resolved, str) and resolved:
            redacted = f"<redacted:reference:{reference}>"
            for variant in _secret_message_variants(resolved):
                if variant and variant != redacted:
                    redactions[variant] = redacted

    visit(value)
    return sorted(redactions.items(), key=lambda item: len(item[0]), reverse=True)


def _secret_message_variants(secret: str) -> set[str]:
    variants = {secret}
    for literal in (repr(secret), ascii(secret), json.dumps(secret), json.dumps(secret, ensure_ascii=True)):
        variants.add(literal)
        if len(literal) >= 2 and literal[0] in {"'", '"'} and literal[-1] == literal[0]:
            variants.add(literal[1:-1])
    variants.add(secret.encode("unicode_escape").decode("ascii"))
    return variants


def _replace_outside_redacted_markers(message: str, needle: str, replacement: str) -> str:
    if not needle:
        return message
    parts = re.split(r"(<redacted:reference:[^>]+>)", message)
    return "".join(
        part if part.startswith("<redacted:reference:") else part.replace(needle, replacement)
        for part in parts
    )


def _apply_agentkit_v2_cli_overrides(payload: dict[str, Any], intent: CLIIntent) -> None:
    if intent.env_provider is None:
        return
    dut_agents = [dut for dut in payload.get("dut_agents") or [] if isinstance(dut, dict)]
    if not dut_agents:
        raise AgentKitV2ValidationError("config.cli_override.not_found env_provider selector matched no dut_agents")
    if len(dut_agents) > 1 and intent.dut_id is None and intent.env_id is None:
        raise AgentKitV2ValidationError("config.cli_override.ambiguous env_provider requires --dut-id or --env-id")

    matched_duts: list[dict[str, Any]] = []
    for dut in dut_agents:
        if intent.dut_id is not None and dut.get("dut_id") != intent.dut_id:
            continue
        env_id = dut.get("env_id")
        if intent.env_id is not None and env_id != intent.env_id:
            continue
        matched_duts.append(dut)

    if not matched_duts:
        raise AgentKitV2ValidationError("config.cli_override.not_found env_provider selector matched no dut_agents")

    target_env_ids: set[str] = set()
    for dut in matched_duts:
        env_id = dut.get("env_id")
        if isinstance(env_id, str):
            target_env_ids.add(env_id)

    matched_env = False
    for env in payload.get("environments") or []:
        if isinstance(env, dict) and env.get("env_id") in target_env_ids:
            env["provider"] = intent.env_provider
            matched_env = True
    if not matched_env:
        raise AgentKitV2ValidationError("config.cli_override.not_found env_provider selector matched no environments")


def _expand_env_references(value: Any, *, redact: bool) -> Any:
    if isinstance(value, dict):
        return {key: _expand_env_references(item, redact=redact) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_references(item, redact=redact) for item in value]
    if not isinstance(value, str):
        return value

    match = _ENV_REF_PATTERN.match(value)
    if not match:
        return value
    reference, operator, operand = match.group(1), match.group(2), match.group(3)
    if redact:
        return f"<redacted:reference:{reference}>"

    env_name = reference[4:] if reference.startswith("ENV.") else reference
    if operator == ":?" and not os.getenv(env_name):
        raise AgentKitV2ValidationError(operand or f"environment variable {env_name} is required")
    resolved = os.getenv(env_name, operand if operator == ":-" and operand is not None else "")
    if isinstance(resolved, str) and resolved.strip():
        for caster in (int, float):
            try:
                return caster(resolved)
            except ValueError:
                pass
    return resolved
