from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

import pytest
import yaml

from gage_eval.config.pipeline_builder import PipelineConfigBuildError
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.schema import SchemaValidationError, normalize_pipeline_payload
from gage_eval.pipeline.step_contracts import clear_step_contract_catalog_cache
from gage_eval.registry import registry


TASK_STEP_NAME = "unit_external_harness_task_step"


def _minimal_payload() -> dict:
    return {
        "metadata": {"name": "external-harness-unit"},
        "datasets": [
            {
                "dataset_id": "harbor_ds",
                "type": "harbor_registry",
                "params": {"ref": "unit/dataset@v1"},
            }
        ],
        "backends": [
            {
                "backend_id": "model_backend",
                "type": "litellm",
                "config": {"model": "gpt-4.1-mini"},
            }
        ],
        "environments": [
            {
                "env_id": "docker_env",
                "provider": "docker",
            }
        ],
        "role_adapters": [
            {
                "adapter_id": "harness",
                "role_type": "external_harness",
                "backend_id": "model_backend",
                "env_id": "docker_env",
                "capabilities": ["task_batch_harness"],
                "params": {
                    "harness": {
                        "launcher": {"mode": "python_subprocess"},
                        "agent": {"kind": "base_agent", "name": "UnitAgent"},
                    }
                },
            }
        ],
        "tasks": [
            {
                "task_id": "harbor_task",
                "dataset_id": "harbor_ds",
                "execution_mode": "task_batch_harness",
                "steps": [{"step": TASK_STEP_NAME}],
            }
        ],
    }


def _normal_sample_payload() -> dict:
    return {
        "datasets": [{"dataset_id": "ds", "loader": "builtin"}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "tasks": [
            {
                "task_id": "sample-task",
                "dataset_id": "ds",
                "steps": [{"step": "inference", "adapter_id": "dut"}],
            }
        ],
    }


def _assert_code(payload: dict, code: str) -> SchemaValidationError:
    with pytest.raises(SchemaValidationError) as exc_info:
        _normalize_with_task_step(payload)
    assert any(error.startswith(f"{code}:") for error in exc_info.value.errors)
    return exc_info.value


@contextmanager
def _task_step_registry():
    clone = registry.clone()
    clone.register(
        "pipeline_steps",
        TASK_STEP_NAME,
        object(),
        desc="unit external harness task step",
        step_kind="task",
    )
    with registry.route_to(clone):
        clear_step_contract_catalog_cache()
        try:
            yield
        finally:
            clear_step_contract_catalog_cache()


def _normalize_with_task_step(payload: dict) -> dict:
    with _task_step_registry():
        return normalize_pipeline_payload(payload)


def _pipeline_config_from_dict_with_task_step(payload: dict) -> PipelineConfig:
    with _task_step_registry():
        return PipelineConfig.from_dict(payload)


@pytest.mark.fast
def test_minimal_task_batch_harness_normalize_passes() -> None:
    normalized = _normalize_with_task_step(_minimal_payload())

    assert normalized["tasks"][0]["execution_mode"] == "task_batch_harness"
    assert normalized["role_adapters"][0]["env_id"] == "docker_env"


@pytest.mark.fast
@pytest.mark.parametrize("top_level_key", ("agents", "prompts", "benchmarks", "dut_agents"))
def test_external_harness_forbids_legacy_top_level_sections(top_level_key: str) -> None:
    payload = _minimal_payload()
    payload[top_level_key] = [{"id": "legacy"}]

    _assert_code(payload, "external_harness.config.forbidden_top_level")


@pytest.mark.fast
def test_task_batch_harness_rejects_sample_step() -> None:
    payload = _minimal_payload()
    payload["tasks"][0]["steps"] = [{"step": "inference", "adapter_id": "harness"}]

    exc = _assert_code(payload, "external_harness.config.invalid_steps")

    assert "actual kind 'sample'" in "\n".join(exc.errors)


@pytest.mark.fast
def test_sample_loop_rejects_task_step() -> None:
    payload = _minimal_payload()
    payload["tasks"][0]["execution_mode"] = "sample_loop"

    with pytest.raises(SchemaValidationError, match="expected kind 'sample'"):
        _normalize_with_task_step(payload)


@pytest.mark.fast
def test_external_harness_rejects_invalid_execution_mode_with_invalid_steps() -> None:
    payload = _minimal_payload()
    payload["tasks"][0]["execution_mode"] = "task_batch"

    _assert_code(payload, "external_harness.config.invalid_steps")


@pytest.mark.fast
@pytest.mark.parametrize("shuffle_key", ("shuffle", "shuffle_seed"))
def test_task_batch_harness_rejects_shuffle_controls(shuffle_key: str) -> None:
    payload = _minimal_payload()
    payload["tasks"][0][shuffle_key] = True if shuffle_key == "shuffle" else 7

    _assert_code(payload, "external_harness.config.invalid_dataset_params")


@pytest.mark.fast
@pytest.mark.parametrize(
    ("field", "value", "message"),
    (
        ("failure_policy", "bad", "unsupported failure_policy"),
        ("support_payload_policy", "bad", "support_payload_policy"),
        ("metric_overrides", [{"metric_id": "missing_metric"}], "missing_metric"),
    ),
)
def test_task_batch_harness_keeps_shared_task_validation(field: str, value: object, message: str) -> None:
    payload = _minimal_payload()
    payload["tasks"][0][field] = value

    with pytest.raises(SchemaValidationError, match=message):
        _normalize_with_task_step(payload)


@pytest.mark.fast
@pytest.mark.parametrize(
    ("loader", "params"),
    (
        ("harbor_registry", {}),
        ("harbor_registry", {"ref": "unit/dataset@v1", "registry_url": "https://example.test/registry.json", "registry_path": "/tmp/registry.json"}),
        ("harbor_local_path", {}),
        ("harbor_local_path", {"path": "/tmp/tasks", "path_kind": "file"}),
        ("harbor_local_path", {"path": "/tmp/tasks", "path_scope": "remote"}),
    ),
)
def test_task_batch_harness_rejects_invalid_dataset_params(loader: str, params: dict) -> None:
    payload = _minimal_payload()
    payload["datasets"][0]["type"] = loader
    payload["datasets"][0]["params"] = params

    _assert_code(payload, "external_harness.config.invalid_dataset_params")


@pytest.mark.fast
def test_external_harness_rejects_missing_env_reference() -> None:
    payload = _minimal_payload()
    payload["role_adapters"][0]["env_id"] = "missing_env"

    _assert_code(payload, "external_harness.config.missing_env_id")


@pytest.mark.fast
def test_external_harness_rejects_local_process_provider_with_invalid_environment_provider() -> None:
    payload = _minimal_payload()
    payload["environments"][0]["provider"] = "local_process"

    _assert_code(payload, "external_harness.config.invalid_environment_provider")


@pytest.mark.fast
def test_external_harness_rejects_duplicate_generation_parameters() -> None:
    payload = _minimal_payload()
    payload["backends"][0]["config"]["generation_parameters"] = {"temperature": 0}
    payload["role_adapters"][0]["params"]["harness"]["agent"]["kwargs"] = {
        "temperature": 0.2
    }

    _assert_code(payload, "external_harness.config.invalid_agent")


@pytest.mark.fast
def test_external_harness_allows_supplemental_model_info_without_conflict() -> None:
    payload = _minimal_payload()
    payload["backends"][0]["config"]["model_info"] = {
        "family": "gpt",
        "limits": {"context": 128000},
    }
    payload["role_adapters"][0]["params"]["harness"]["agent"]["kwargs"] = {
        "model_info": {"family": "gpt", "limits": {"output": 4096}, "supports_tools": True}
    }

    normalized = _normalize_with_task_step(payload)

    assert normalized["backends"][0]["config"]["model_info"]["family"] == "gpt"


@pytest.mark.fast
def test_external_harness_rejects_model_info_conflict() -> None:
    payload = _minimal_payload()
    payload["backends"][0]["config"]["model_info"] = {"family": "gpt"}
    payload["role_adapters"][0]["params"]["harness"]["agent"]["kwargs"] = {
        "model_info": {"family": "claude"}
    }

    _assert_code(payload, "external_harness.translate.model_info_conflict")


@pytest.mark.fast
def test_external_harness_rejects_agent_env_with_invalid_agent() -> None:
    payload = _minimal_payload()
    payload["role_adapters"][0]["params"]["harness"]["agent"]["env"] = {"TOKEN": "x"}

    _assert_code(payload, "external_harness.config.invalid_agent")


@pytest.mark.fast
def test_pipeline_config_from_dict_propagates_external_harness_fields() -> None:
    payload = _minimal_payload()
    payload["role_adapters"][0]["trial_policy"] = {"trials": 2}

    config = _pipeline_config_from_dict_with_task_step(payload)

    assert config.datasets[0].loader == "harbor_registry"
    assert config.role_adapters[0].env_id == "docker_env"
    assert config.role_adapters[0].trial_policy == {"trials": 2}
    assert config.tasks[0].execution_mode == "task_batch_harness"


@pytest.mark.fast
def test_pipeline_config_from_dict_forbids_external_harness_top_level_sections() -> None:
    payload = _minimal_payload()
    payload["agents"] = [
        {
            "agent_id": "legacy_agent",
            "scheduler": {"type": "framework_loop", "backend_id": "model_backend"},
            "config": {},
        }
    ]
    payload["benchmarks"] = [{"benchmark_id": "legacy_bench", "kit_id": "legacy", "config": {}}]
    payload["dut_agents"] = [
        {
            "dut_id": "legacy_dut",
            "agent_id": "legacy_agent",
            "env_id": "docker_env",
            "benchmark_id": "legacy_bench",
        }
    ]

    with pytest.raises(PipelineConfigBuildError) as exc_info:
        _pipeline_config_from_dict_with_task_step(payload)

    assert "external_harness.config.forbidden_top_level" in str(exc_info.value)


@pytest.mark.fast
def test_pipeline_config_from_dict_defaults_new_fields() -> None:
    config = PipelineConfig.from_dict(_normal_sample_payload())

    assert config.role_adapters[0].env_id is None
    assert config.role_adapters[0].trial_policy is None
    assert config.tasks[0].execution_mode == "sample_loop"


@pytest.mark.fast
@pytest.mark.parametrize(
    "fixture_name",
    ("tau2_local_minimal.yaml", "swebench_docker.yaml"),
)
def test_agentkit_v2_fixture_compatibility_defaults_new_fields(fixture_name: str) -> None:
    fixture = Path("tests/fixtures/agentkit_v2") / fixture_name
    payload = yaml.safe_load(fixture.read_text())

    config = PipelineConfig.from_dict(payload)

    assert config.role_adapters
    assert config.role_adapters[0].env_id is None
    assert config.role_adapters[0].trial_policy is None


COVERED_CONFIG_CODES = {
    "external_harness.config.forbidden_top_level",
    "external_harness.config.invalid_agent",
    "external_harness.config.invalid_dataset_params",
    "external_harness.config.invalid_steps",
    "external_harness.config.missing_env_id",
}

LATER_TASK_CONFIG_CODES = {
    "external_harness.config.invalid_concurrency": "covered by bridge/runtime concurrency tests in a later task",
    "external_harness.config.invalid_environment_override": "legacy environment override coverage belongs with translation bridge hardening",
    "external_harness.config.invalid_environment_provider": "provider matrix coverage belongs with environment bridge tests",
    "external_harness.config.invalid_launcher": "launcher execution coverage belongs with Task 04 bridge work",
    "external_harness.config.invalid_loader": "Harbor loader matrix coverage belongs with Task 03",
    "external_harness.config.missing_backend_id": "backend resolution coverage belongs with adapter translation tests",
    "external_harness.config.provider_mismatch": "provider override matrix coverage belongs with environment bridge tests",
    "external_harness.config.secret_agent_env_forbidden": "secret handling coverage belongs with security-focused bridge tests",
    "external_harness.config.unknown_adapter": "step adapter routing coverage belongs with task step integration tests",
}


@pytest.mark.fast
def test_appendix_a_external_harness_config_code_coverage_marker() -> None:
    appendix_a_config_codes = {
        "external_harness.config.forbidden_top_level",
        "external_harness.config.invalid_agent",
        "external_harness.config.invalid_concurrency",
        "external_harness.config.invalid_dataset_params",
        "external_harness.config.invalid_environment_override",
        "external_harness.config.invalid_environment_provider",
        "external_harness.config.invalid_launcher",
        "external_harness.config.invalid_loader",
        "external_harness.config.invalid_steps",
        "external_harness.config.missing_backend_id",
        "external_harness.config.missing_env_id",
        "external_harness.config.provider_mismatch",
        "external_harness.config.secret_agent_env_forbidden",
        "external_harness.config.unknown_adapter",
    }

    assert appendix_a_config_codes == COVERED_CONFIG_CODES | set(LATER_TASK_CONFIG_CODES)
    assert all(LATER_TASK_CONFIG_CODES.values())
