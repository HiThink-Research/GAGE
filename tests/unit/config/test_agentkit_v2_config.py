from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from gage_eval.agent_eval_kits import load_benchmark_kit
from gage_eval.agent_eval_kits.common import EmptyKitConfig, BenchmarkKitEntry, validate_benchmark_kit_entry
from gage_eval.agent_runtime.compiled_plan import CompiledRuntimePlan
from gage_eval.agent_runtime.schedulers.framework_loop import StaticModelBackendAdapter
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry
from gage_eval.config.agentkit_v2 import (
    AgentKitV2ValidationError,
    AgentkitV2ConfigModel,
    build_agentkit_v2_runtime_bindings,
    load_agentkit_v2_config_payload,
    materialize_agentkit_v2_config_payload,
    materialize_agentkit_v2_runtime_config_payload,
    resolve_agentkit_v2_runtime_binding_specs,
)
from gage_eval.config.loader_cli import CLIIntent


FIXTURE_DIR = Path(__file__).resolve().parents[2] / "fixtures" / "agentkit_v2"


class StrictKitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


class LooseKitConfig(BaseModel):
    model_config = ConfigDict(extra="allow")


class _McpClient:
    def list_tools(self) -> list[dict[str, Any]]:
        return [{"name": "mcp_lookup", "inputSchema": {"type": "object"}}]


def _load_fixture(name: str) -> dict[str, Any]:
    return load_agentkit_v2_config_payload(FIXTURE_DIR / name)


def _minimal_payload() -> dict[str, Any]:
    return {
        "kind": "PipelineConfig",
        "metadata": {"name": "minimal"},
        "backends": [
            {
                "backend_id": "model",
                "type": "litellm",
                "config": {"model": "gpt-4.1-mini"},
            }
        ],
        "agents": [
            {
                "agent_id": "agent",
                "scheduler": {"type": "framework_loop", "backend_id": "model"},
                "config": {},
            }
        ],
        "benchmarks": [
            {"benchmark_id": "bench", "kit_id": "tau2", "config": {}},
        ],
        "environments": [
            {
                "env_id": "env",
                "provider": "local_process",
                "profile_id": "tau2_local",
                "profile": {"asset_dir": "tests/fixtures/agentkit_v2/tau2"},
            }
        ],
        "dut_agents": [
            {
                "dut_id": "dut",
                "agent_id": "agent",
                "env_id": "env",
                "benchmark_id": "bench",
            }
        ],
    }


def _kit_entry(**overrides: Any) -> BenchmarkKitEntry:
    def workflow_resolver(scheduler_type: str) -> object:
        if scheduler_type not in {"framework_loop", "installed_client"}:
            raise KeyError(scheduler_type)
        return object()

    defaults: dict[str, Any] = {
        "kit_id": "demo",
        "config_schema": StrictKitConfig,
        "default_environment_provider": "local_process",
        "default_environment_profile_by_provider": {"local_process": "local"},
        "environment_profiles": {"local": {"asset_dir": "assets/demo"}},
        "verifier_environment_policy": "kit_judge",
        "verifier_environment_profile_id": "local",
        "supported_schedulers": ("framework_loop",),
        "workflow_resolver": workflow_resolver,
        "tool_registry_factory": lambda: object(),
        "verifier_adapter_factory": lambda: object(),
        "artifact_manifest_factory": lambda: {},
    }
    defaults.update(overrides)
    return BenchmarkKitEntry(**defaults)


@pytest.mark.fast
def test_valid_tau2_local_process_minimal_yaml_expands_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")

    payload = _load_fixture("tau2_local_minimal.yaml")

    assert payload["trial_policy"] == {"trials": 1}
    assert payload["dut_agents"][0]["trial_policy"] == {"trials": 1}
    assert payload["environments"][0]["provider"] == "local_process"
    assert payload["environments"][0]["lifecycle"] == "per_sample"
    assert payload["effective_config"]["backends"][0]["config"]["api_key"] == (
        "<redacted:reference:ENV.MODEL_API_KEY>"
    )


@pytest.mark.fast
def test_materialized_payload_repr_redacts_env_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")

    payload = _load_fixture("tau2_local_minimal.yaml")

    assert payload["backends"][0]["config"]["api_key"] == "<redacted:reference:ENV.MODEL_API_KEY>"
    assert "secret-value" not in repr(payload)
    assert "<redacted:reference:ENV.MODEL_API_KEY>" in repr(payload)


@pytest.mark.fast
def test_materialized_payload_repr_redacts_bare_env_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "bare-secret-value")
    payload = _minimal_payload()
    payload["backends"][0]["config"]["api_key"] = "${MODEL_API_KEY}"

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)

    assert materialized["backends"][0]["config"]["api_key"] == "<redacted:reference:MODEL_API_KEY>"
    assert materialized["effective_config"]["backends"][0]["config"]["api_key"] == (
        "<redacted:reference:MODEL_API_KEY>"
    )
    assert "bare-secret-value" not in repr(materialized)


@pytest.mark.fast
def test_materialized_payload_redacts_hardcoded_secret_keynames() -> None:
    payload = _minimal_payload()
    payload["backends"][0]["config"]["api_key"] = "dummy"
    payload["benchmarks"][0]["config"]["user_simulator"] = {
        "model": "openai/qwen/qwen3.5-9b",
        "model_args": {"api_key": "hardcoded-user-key"},
    }

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    assert materialized["backends"][0]["config"]["api_key"] == "<redacted:keyname:api_key>"
    assert (
        materialized["effective_config"]["benchmarks"][0]["config"]["user_simulator"]["model_args"]["api_key"]
        == "<redacted:keyname:api_key>"
    )
    assert "hardcoded-user-key" not in repr(materialized)
    assert runtime_config["backends"][0]["config"]["api_key"] == "dummy"
    assert runtime_config["benchmarks"][0]["config"]["user_simulator"]["model_args"]["api_key"] == "hardcoded-user-key"


@pytest.mark.fast
def test_materialized_payload_preserves_token_usage_fields_while_redacting_secret_tokens() -> None:
    payload = _minimal_payload()
    payload["backends"][0]["config"].update(
        {
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "total_tokens": 579,
            "completion_tokens_details": {"reasoning_tokens": 7},
            "agent_total_tokens": 579,
            "access_token": "access-real",
            "session_token": "session-real",
            "token": "plain-token-real",
        }
    )

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    backend_config = materialized["backends"][0]["config"]
    assert backend_config["prompt_tokens"] == 123
    assert backend_config["completion_tokens"] == 456
    assert backend_config["total_tokens"] == 579
    assert backend_config["completion_tokens_details"]["reasoning_tokens"] == 7
    assert backend_config["agent_total_tokens"] == 579
    assert backend_config["access_token"] == "<redacted:keyname:access_token>"
    assert backend_config["session_token"] == "<redacted:keyname:session_token>"
    assert backend_config["token"] == "<redacted:keyname:token>"
    assert runtime_config["backends"][0]["config"]["access_token"] == "access-real"
    assert runtime_config["backends"][0]["config"]["prompt_tokens"] == 123


@pytest.mark.fast
@pytest.mark.parametrize(
    ("placeholder", "redacted"),
    [
        ("${MODEL_API_KEY}", "<redacted:reference:MODEL_API_KEY>"),
        ("${ENV.MODEL_API_KEY}", "<redacted:reference:ENV.MODEL_API_KEY>"),
    ],
)
def test_validation_error_redacts_env_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    placeholder: str,
    redacted: str,
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "validation-secret-value")
    payload = _minimal_payload()
    payload["environments"][0]["provider"] = placeholder

    with pytest.raises(AgentKitV2ValidationError) as excinfo:
        materialize_agentkit_v2_config_payload(payload, source_path=None)

    message = str(excinfo.value)
    assert "validation-secret-value" not in message
    assert redacted in message
    assert "environments.0.provider" in message


@pytest.mark.fast
@pytest.mark.parametrize(
    ("secret", "escaped_fragments"),
    [
        ("line1\nline2", ("line1\\nline2",)),
        ("tab\tvalue", ("tab\\tvalue",)),
        ("abc\\def", ("abc\\\\def",)),
    ],
)
def test_validation_error_redacts_escaped_env_secret_values(
    monkeypatch: pytest.MonkeyPatch,
    secret: str,
    escaped_fragments: tuple[str, ...],
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", secret)
    payload = _minimal_payload()
    payload["environments"][0]["provider"] = "${MODEL_API_KEY}"

    with pytest.raises(AgentKitV2ValidationError) as excinfo:
        materialize_agentkit_v2_config_payload(payload, source_path=None)

    message = str(excinfo.value)
    assert secret not in message
    for fragment in escaped_fragments:
        assert fragment not in message
    assert "<redacted:reference:MODEL_API_KEY>" in message
    assert "environments.0.provider" in message


@pytest.mark.fast
def test_valid_swebench_docker_yaml_expands_trial_policy_trials_3() -> None:
    payload = _load_fixture("swebench_docker.yaml")

    assert payload["trial_policy"] == {"trials": 3}
    assert payload["dut_agents"][0]["trial_policy"] == {"trials": 3}
    assert payload["environments"][0]["provider"] == "docker"


@pytest.mark.fast
def test_agent_scheduler_backend_id_reference_missing_raises() -> None:
    payload = _minimal_payload()
    payload["agents"][0]["scheduler"]["backend_id"] = "missing"

    with pytest.raises(AgentKitV2ValidationError, match="config.reference.missing"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_environment_profile_provider_config_alias_is_rejected() -> None:
    payload = _minimal_payload()
    payload["environments"][0]["profile"]["provider_config"] = {"workdir": "/workspace"}

    with pytest.raises(AgentKitV2ValidationError, match="environments.env.profile.provider_config"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_framework_loop_agent_scheduler_backend_id_is_required() -> None:
    payload = _minimal_payload()
    payload["agents"][0]["scheduler"].pop("backend_id")

    with pytest.raises(AgentKitV2ValidationError, match="scheduler.backend_id.required"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_agentkit_v2_runtime_binding_wraps_scheduler_backend_id_static_backend() -> None:
    payload = _minimal_payload()
    static_backend = object()
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": static_backend},
    )

    binding = bindings["dut"]
    scheduler = binding.executor_ref.compiled_plan.scheduler_handle
    assert binding.backend_id == "model"
    assert isinstance(scheduler._backend, StaticModelBackendAdapter)
    assert scheduler._backend.static_backend is static_backend


@pytest.mark.fast
def test_agentkit_v2_agent_tooling_resolves_mcp_and_skills_into_runtime_registry() -> None:
    payload = _minimal_payload()
    payload["agents"][0]["tooling"] = {
        "mcp_servers": ["local_mcp"],
        "skill_ids": ["local_skill"],
        "skill_manifests": {
            "local_skill": {
                "tools": [{"name": "skill_lookup", "inputSchema": {"type": "object"}}],
            }
        },
    }
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": object()},
        mcp_clients={"local_mcp": _McpClient()},
    )

    registry = bindings["dut"].executor_ref.compiled_plan.tool_registry
    assert isinstance(registry, RuntimeToolRegistry)
    assert materialized["agents"][0]["tooling"]["skill_ids"] == ["local_skill"]
    assert registry.get("mcp_lookup").provider_kind == "mcp"
    assert registry.get("skill_lookup").schema.metadata["provider"] == "skill:local_skill"


@pytest.mark.fast
def test_agentkit_v2_runtime_binding_uses_environment_provider_as_resource_kind() -> None:
    payload = _minimal_payload()
    payload["environments"][0]["provider"] = "docker"
    static_backend = object()
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)

    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": static_backend},
    )

    plan = bindings["dut"].executor_ref.compiled_plan
    assert bindings["dut"].environment_provider == "docker"
    assert plan.environment_provider == "docker"
    assert plan.resource_plan["resource_kind"] == "docker"
    assert plan.resource_plan["environment_profile"]["provider"] == "docker"


@pytest.mark.fast
def test_agentkit_v2_runtime_binding_requires_private_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_SECRET", "runtime-secret-value")
    payload = _minimal_payload()
    payload["environments"][0]["profile"]["token"] = "${ENV.SANDBOX_SECRET}"
    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)

    with pytest.raises(AgentKitV2ValidationError, match="runtime_config.required"):
        resolve_agentkit_v2_runtime_binding_specs(materialized)
    with pytest.raises(AgentKitV2ValidationError, match="runtime_config.required"):
        build_agentkit_v2_runtime_bindings(
            materialized,
            backends={"model": object()},
        )


@pytest.mark.fast
def test_agentkit_v2_runtime_binding_uses_unredacted_environment_runtime_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SANDBOX_SECRET", "runtime-secret-value")
    payload = _minimal_payload()
    payload["environments"][0]["profile"]["token"] = "${ENV.SANDBOX_SECRET}"
    payload["environments"][0]["provider_config"] = {}
    payload["environments"][0]["startup_env"] = {}
    payload["environments"][0]["resources"] = {}
    payload["environments"][0]["provider_config"]["api_key"] = "${ENV.SANDBOX_SECRET}"
    payload["environments"][0]["startup_env"]["SECRET_ENV"] = "${ENV.SANDBOX_SECRET}"
    payload["environments"][0]["resources"]["secret_mount"] = "${ENV.SANDBOX_SECRET}"

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)
    specs = resolve_agentkit_v2_runtime_binding_specs(
        materialized,
        runtime_config=runtime_config,
    )
    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": object()},
    )

    assert "runtime-secret-value" not in repr(materialized)
    assert materialized["environments"][0]["profile"]["token"] == (
        "<redacted:reference:ENV.SANDBOX_SECRET>"
    )
    assert materialized["effective_config"]["environments"][0]["startup_env"]["SECRET_ENV"] == (
        "<redacted:reference:ENV.SANDBOX_SECRET>"
    )
    assert specs["dut"].environment_profile["token"] == "runtime-secret-value"
    assert specs["dut"].provider_config["api_key"] == "runtime-secret-value"
    assert specs["dut"].startup_env["SECRET_ENV"] == "runtime-secret-value"
    assert specs["dut"].resources["secret_mount"] == "runtime-secret-value"
    assert bindings["dut"].executor_ref.compiled_plan.resource_plan["provider_config"]["api_key"] == (
        "runtime-secret-value"
    )


@pytest.mark.fast
def test_agentkit_v2_runtime_binding_keeps_tau2_runtime_config_out_of_provider_config() -> None:
    payload = _minimal_payload()
    payload["benchmarks"][0]["config"] = {
        "domain": "telecom",
        "data_dir": "/tmp/tau2",
        "max_steps": 7,
        "max_errors": 2,
        "respond_tool_name": "respond",
        "user_simulator": {
            "model": "openai/qwen",
            "model_args": {"api_key": "dummy"},
        },
    }
    payload["environments"][0]["provider_config"] = {}

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    runtime_config = materialize_agentkit_v2_runtime_config_payload(payload, source_path=None)
    specs = resolve_agentkit_v2_runtime_binding_specs(
        materialized,
        runtime_config=runtime_config,
    )
    bindings = build_agentkit_v2_runtime_bindings(
        materialized,
        runtime_config=runtime_config,
        backends={"model": object()},
    )

    spec = specs["dut"]
    plan = bindings["dut"].executor_ref.compiled_plan
    assert spec.provider_config == {}
    assert spec.benchmark_config["data_dir"] == "/tmp/tau2"
    assert spec.benchmark_config["max_steps"] == 7
    assert spec.benchmark_config["user_simulator"]["model"] == "openai/qwen"
    assert plan.provider_config == {}
    assert plan.kit_config["data_dir"] == "/tmp/tau2"
    assert plan.resource_plan["provider_config"] == {}


@pytest.mark.fast
def test_agent_scheduler_type_defaults_to_framework_loop() -> None:
    payload = _minimal_payload()
    payload["agents"][0]["scheduler"].pop("type")

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)

    assert materialized["agents"][0]["scheduler"]["type"] == "framework_loop"


@pytest.mark.fast
def test_effective_config_includes_defaults_and_redacts_secrets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")
    payload = _minimal_payload()
    payload["backends"][0]["config"]["api_key"] = "${MODEL_API_KEY}"
    payload["agents"][0]["scheduler"].pop("type")
    payload["environments"][0].pop("lifecycle", None)

    materialized = materialize_agentkit_v2_config_payload(payload, source_path=None)
    effective = materialized["effective_config"]

    assert materialized["agents"][0]["scheduler"]["type"] == "framework_loop"
    assert materialized["environments"][0]["lifecycle"] == "per_sample"
    assert effective["agents"][0]["scheduler"]["type"] == "framework_loop"
    assert effective["environments"][0]["lifecycle"] == "per_sample"
    assert materialized["backends"][0]["config"]["api_key"] == "<redacted:reference:MODEL_API_KEY>"
    assert effective["backends"][0]["config"]["api_key"] == "<redacted:reference:MODEL_API_KEY>"
    assert "secret-value" not in repr(materialized)
    assert "effective_config" not in effective


@pytest.mark.fast
@pytest.mark.parametrize(
    ("field_name", "missing_id"),
    [
        ("agent_id", "missing_agent"),
        ("env_id", "missing_env"),
        ("benchmark_id", "missing_benchmark"),
    ],
)
def test_dut_agent_references_missing_raise(field_name: str, missing_id: str) -> None:
    payload = _minimal_payload()
    payload["dut_agents"][0][field_name] = missing_id

    with pytest.raises(AgentKitV2ValidationError, match="config.reference.missing"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
@pytest.mark.parametrize(
    "legacy_key",
    [
        "schema_version",
        "runtime_version",
        "agent_backends",
        "agent_backend_id",
        "benchmark_configs",
        "sandbox_profiles",
        "sandbox_profile_id",
        "kit",
        "scheduler",
        "environment",
    ],
)
def test_agentkit_v2_legacy_keys_fail_fast(legacy_key: str) -> None:
    payload = _minimal_payload()
    payload[legacy_key] = "legacy"

    with pytest.raises(AgentKitV2ValidationError, match=f"config.legacy_key.{legacy_key}"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_compiled_runtime_plan_carries_all_required_fields() -> None:
    names = {field.name for field in fields(CompiledRuntimePlan)}

    assert {
        "run_id",
        "dut_id",
        "agent_id",
        "env_id",
        "benchmark_id",
        "trial_policy",
        "kit_id",
        "kit_entry",
        "kit_config",
        "agent_config",
        "scheduler_type",
        "scheduler_config",
        "environment_provider",
        "environment_profile_id",
        "environment_profile",
        "lifecycle",
        "provider_config",
        "startup_env",
        "resources",
        "verifier_environment_policy",
        "verifier_environment_profile_id",
        "workflow_bundle",
        "tool_registry",
        "tool_provider_adapter",
        "verifier_adapter",
        "artifact_sink",
    }.issubset(names)


@pytest.mark.fast
def test_compiled_runtime_plan_excludes_legacy_fields() -> None:
    names = {field.name for field in fields(CompiledRuntimePlan)}

    assert names.isdisjoint(
        {
            "agent_backend_id",
            "agent_backends",
            "AgentBackendSpec",
            "benchmark_configs",
            "sandbox_profile_id",
            "provider_sdk_object",
        }
    )


@pytest.mark.fast
def test_benchmark_kit_entry_carries_required_fields_and_no_runtime_version() -> None:
    names = {field.name for field in fields(BenchmarkKitEntry)}
    entry = _kit_entry()

    assert {
        "kit_id",
        "config_schema",
        "default_environment_provider",
        "default_environment_profile_by_provider",
        "environment_profiles",
        "verifier_environment_policy",
        "verifier_environment_profile_id",
        "supported_schedulers",
        "workflow_resolver",
        "tool_registry_factory",
        "verifier_adapter_factory",
        "artifact_manifest_factory",
    }.issubset(names)
    assert {
        "benchmark_kit_id",
        "verifier_kind",
        "resource_requirements",
        "lifecycle_policy",
        "state_schema_keys",
        "verifier_resource_resolver",
        "trace_mapper",
    }.isdisjoint(names)
    assert "runtime_version" not in names
    assert not hasattr(entry, "benchmark_kit_id")
    assert not hasattr(entry, "runtime_version")


@pytest.mark.fast
@pytest.mark.parametrize("kit_id", ["appworld", "swebench", "tau2"])
def test_builtin_kit_entries_do_not_expose_runtime_version(kit_id: str) -> None:
    loaded_entry = load_benchmark_kit(kit_id)

    assert not hasattr(loaded_entry, "runtime_version")
    assert not hasattr(loaded_entry.runtime_entry, "runtime_version")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"config_schema": object}, "config_schema"),
        ({"config_schema": LooseKitConfig}, "config_schema.extra_forbid"),
        ({"default_environment_provider": "docker"}, "default_environment_provider"),
        (
            {"default_environment_profile_by_provider": {"local_process": "missing"}},
            "default_environment_profile",
        ),
        ({"environment_profiles": {"local": {}}}, "environment_profiles.local.asset_dir"),
        ({"verifier_environment_policy": "external"}, "verifier_environment_policy"),
        (
            {
                "verifier_environment_policy": "kit_judge",
                "verifier_environment_profile_id": "missing",
            },
            "verifier_environment_profile_id",
        ),
        ({"supported_schedulers": ("missing",)}, "supported_schedulers.missing.workflow"),
        ({"tool_registry_factory": None}, "framework_loop.tool_registry"),
        ({"verifier_adapter_factory": None}, "verifier_adapter_factory"),
        ({"artifact_manifest_factory": None}, "artifact_manifest_factory"),
    ],
)
def test_kit_registry_loading_validation_rejects_invalid_entries(
    overrides: dict[str, Any],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_benchmark_kit_entry(_kit_entry(**overrides))


@pytest.mark.fast
def test_kit_registry_loading_rejects_missing_explicit_config_schema() -> None:
    with pytest.raises(ValueError, match="config_schema"):
        validate_benchmark_kit_entry(
            BenchmarkKitEntry(
                kit_id="missing_schema",
                default_environment_provider="local_process",
                default_environment_profile_by_provider={"local_process": "local"},
                environment_profiles={"local": {"asset_dir": "assets/demo"}},
                verifier_environment_policy="kit_judge",
                verifier_environment_profile_id="local",
                supported_schedulers=("framework_loop",),
                workflow_resolver=lambda scheduler_type: object(),
                tool_registry_factory=lambda: object(),
                verifier_adapter_factory=lambda: object(),
                artifact_manifest_factory=lambda: {},
            )
        )


@pytest.mark.fast
@pytest.mark.parametrize("kit_id", ["appworld", "swebench", "tau2"])
def test_builtin_kit_entries_have_explicit_extra_forbid_config_schema(kit_id: str) -> None:
    loaded_entry = load_benchmark_kit(kit_id)

    assert loaded_entry.config_schema is not EmptyKitConfig
    assert loaded_entry.config_schema.model_config.get("extra") == "forbid"


@pytest.mark.fast
def test_benchmark_config_unknown_field_fails_fast() -> None:
    payload = _minimal_payload()
    payload["benchmarks"][0]["config"]["unknown_field"] = "nope"

    with pytest.raises(AgentKitV2ValidationError, match="config.kit_schema.unknown_field"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_multiple_dut_agents_env_provider_override_requires_dut_or_env_id() -> None:
    payload = _minimal_payload()
    payload["environments"].append(
        {
            "env_id": "env2",
            "provider": "docker",
            "profile_id": "tau2_docker",
            "profile": {"asset_dir": "tests/fixtures/agentkit_v2/tau2-docker"},
        }
    )
    payload["dut_agents"].append(
        {
            "dut_id": "dut2",
            "agent_id": "agent",
            "env_id": "env2",
            "benchmark_id": "bench",
        }
    )

    with pytest.raises(AgentKitV2ValidationError, match="config.cli_override.ambiguous"):
        materialize_agentkit_v2_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(env_provider="local_process"),
        )


@pytest.mark.fast
def test_env_provider_override_missing_dut_selector_raises() -> None:
    payload = _minimal_payload()

    with pytest.raises(AgentKitV2ValidationError, match="config.cli_override.not_found"):
        materialize_agentkit_v2_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(env_provider="docker", dut_id="missing_dut"),
        )


@pytest.mark.fast
def test_env_provider_override_mismatched_dut_and_env_selectors_raises() -> None:
    payload = _minimal_payload()
    payload["environments"].append(
        {
            "env_id": "env2",
            "provider": "local_process",
            "profile_id": "tau2_local_2",
            "profile": {"asset_dir": "tests/fixtures/agentkit_v2/tau2-2"},
        }
    )
    payload["dut_agents"].append(
        {
            "dut_id": "dut2",
            "agent_id": "agent",
            "env_id": "env2",
            "benchmark_id": "bench",
        }
    )

    with pytest.raises(AgentKitV2ValidationError, match="config.cli_override.not_found"):
        materialize_agentkit_v2_config_payload(
            payload,
            source_path=None,
            cli_intent=CLIIntent(env_provider="docker", dut_id="dut", env_id="env2"),
        )


@pytest.mark.fast
def test_env_provider_override_is_reflected_in_effective_config() -> None:
    payload = _minimal_payload()

    materialized = materialize_agentkit_v2_config_payload(
        payload,
        source_path=None,
        cli_intent=CLIIntent(env_provider="docker", dut_id="dut"),
    )

    assert materialized["environments"][0]["provider"] == "docker"
    assert materialized["effective_config"]["environments"][0]["provider"] == "docker"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("section", "id_field"),
    [
        ("backends", "backend_id"),
        ("agents", "agent_id"),
        ("benchmarks", "benchmark_id"),
        ("environments", "env_id"),
    ],
)
def test_agentkit_v2_duplicate_ids_raise(section: str, id_field: str) -> None:
    payload = _minimal_payload()
    payload[section].append(dict(payload[section][0]))

    with pytest.raises(AgentKitV2ValidationError, match=f"config.duplicate_id.{section}.{id_field}"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_agentkit_v2_duplicate_dut_ids_raise() -> None:
    payload = _minimal_payload()
    payload["dut_agents"].append(dict(payload["dut_agents"][0]))

    with pytest.raises(AgentKitV2ValidationError, match="config.duplicate_id.dut_agents.dut_id"):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_env_secret_reference_in_effective_config_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MODEL_API_KEY", "secret-value")

    payload = _load_fixture("tau2_local_minimal.yaml")

    assert payload["effective_config"]["backends"][0]["config"]["api_key"] == (
        "<redacted:reference:ENV.MODEL_API_KEY>"
    )
    assert "secret-value" not in repr(payload["effective_config"])


@pytest.mark.fast
@pytest.mark.parametrize(
    ("lifecycle", "match"),
    [
        ("per_task", "config.environment.lifecycle.per_task"),
        ("per_run", "config.environment.lifecycle.unsupported"),
        ("forever", "config.environment.lifecycle.unsupported"),
    ],
)
def test_environments_lifecycle_non_per_sample_fails_fast(lifecycle: str, match: str) -> None:
    payload = _minimal_payload()
    payload["environments"][0]["lifecycle"] = lifecycle

    with pytest.raises(AgentKitV2ValidationError, match=match):
        materialize_agentkit_v2_config_payload(payload, source_path=None)


@pytest.mark.fast
def test_agentkit_v2_core_model_forbids_extra_fields() -> None:
    extra = AgentkitV2ConfigModel.model_config.get("extra")

    assert extra == "forbid"
