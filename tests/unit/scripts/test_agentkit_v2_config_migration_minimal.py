from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from gage_eval.config.agentkit_v2 import load_agentkit_v2_config_payload


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "scripts" / "migrate_agentkit_v1_config_to_v2.py"
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "agentkit_v2" / "migration"


def _load_migration_module() -> Any:
    spec = importlib.util.spec_from_file_location("migrate_agentkit_v1_config_to_v2", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    assert isinstance(payload, dict)
    return payload


def _run_migration_cli_paths(
    input_path: Path,
    output_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
        env=env,
    )


def _run_migration_cli(
    input_name: str,
    output_path: Path,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return _run_migration_cli_paths(FIXTURE_DIR / input_name, output_path, env=env)


def test_migrates_agent_backends_and_role_adapter_to_v2_loader_payload(tmp_path: Path) -> None:
    module = _load_migration_module()
    output_path = tmp_path / "migrated.yaml"

    result = module.migrate_file(FIXTURE_DIR / "v1_role_adapter_minimal.yaml", output_path)

    assert result.ok
    migrated = _read_yaml(output_path)
    assert migrated["kind"] == "AgentEvalConfig"
    assert migrated["backends"] == [
        {
            "backend_id": "tau2_openai_http",
            "type": "litellm",
            "config": {"model": "gpt-4.1-mini"},
        }
    ]
    assert migrated["agents"] == [
        {
            "agent_id": "tau2_agent",
            "scheduler": {
                "type": "framework_loop",
                "backend_id": "tau2_openai_http",
            },
            "config": {},
        }
    ]
    assert migrated["benchmarks"] == [
        {
            "benchmark_id": "tau2_benchmark",
            "kit_id": "tau2",
            "config": {},
        }
    ]
    assert migrated["environments"][0]["provider"] == "local_process"
    assert migrated["dut_agents"] == [
        {
            "dut_id": "tau2_agent_dut",
            "agent_id": "tau2_agent",
            "env_id": "tau2_env",
            "benchmark_id": "tau2_benchmark",
        }
    ]
    assert load_agentkit_v2_config_payload(output_path)["agents"][0]["scheduler"]["backend_id"] == "tau2_openai_http"


def test_migrates_legacy_scheduler_model_to_generated_backend_id(tmp_path: Path) -> None:
    module = _load_migration_module()
    output_path = tmp_path / "migrated.yaml"

    result = module.migrate_file(FIXTURE_DIR / "legacy_scheduler_model_minimal.yaml", output_path)

    assert result.ok
    migrated = _read_yaml(output_path)
    assert migrated["backends"] == [
        {
            "backend_id": "inline_agent_model",
            "type": "litellm",
            "config": {"model": "gpt-4.1-mini"},
        }
    ]
    assert migrated["agents"][0]["scheduler"] == {
        "type": "framework_loop",
        "backend_id": "inline_agent_model",
    }
    assert "model" not in migrated["agents"][0]["scheduler"]
    assert load_agentkit_v2_config_payload(output_path)["backends"][0]["backend_id"] == "inline_agent_model"


def test_cli_reports_unmapped_fields_and_exits_nonzero(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_unmapped_fields.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "datasets" in completed.stderr
    assert "tasks" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_agent_backend_config_that_would_be_dropped(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_agent_backend_unsupported_config.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agent_backends[0].config" in completed.stderr
    assert "force_tool_choice" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_inline_agent_backend_config_that_would_be_misclassified(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_inline_agent_backend_unsupported_config.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agent_backends[0].config" in completed.stderr
    assert "force_tool_choice" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_top_level_scheduler_model_ambiguous_with_agent_backend_id(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_top_level_scheduler_model_ambiguous.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "scheduler.model" in completed.stderr
    assert "agent_backend_id" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_legacy_agent_scheduler_backend_id_missing_from_migrated_backends(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("legacy_agent_missing_scheduler_backend.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agents.missing_ref_agent.scheduler.backend_id=missing_backend" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_dut_role_adapter_agent_backend_id_missing_from_migrated_backends(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_missing_backend_ref.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters[1]" in completed.stderr
    assert "missing_agent" in completed.stderr
    assert "missing_backend" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_top_level_agent_backend_id_missing_even_with_scheduler_model(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_top_level_agent_backend_id_unresolved_with_model.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "top-level agent_backend_id=missing_backend" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_sanitized_role_adapter_agent_id_collision(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_sanitized_agent_id_collision.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "sanitized agent_id collision" in completed.stderr
    assert "agent-1" in completed.stderr
    assert "agent_1" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_unreferenced_agent_backend_that_cannot_migrate(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_unreferenced_agent_backend_unknown_backend.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agent_backends[1]" in completed.stderr
    assert "unused_backend" in completed.stderr
    assert "missing_model" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_inline_agent_backend_sanitized_backend_id_collision(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_inline_agent_backend_sanitized_backend_id_collision.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "sanitized backend_id collision" in completed.stderr
    assert "a-b" in completed.stderr
    assert "a_b" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_invalid_top_level_scheduler_type(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_top_level_scheduler_invalid_type.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "scheduler.type=bad" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_duplicate_static_backend_ids(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_duplicate_static_backends.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "duplicate backends.backend_id=model" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_duplicate_agent_backend_ids(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_duplicate_agent_backends.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "duplicate agent_backends.agent_backend_id=agent_backend" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_unsupported_agent_backend_type(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_agent_backend_unsupported_type.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agent_backends[0].type=agent_class" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_unknown_role_adapter_runtime_id(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_unknown_runtime_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters[0]" in completed.stderr
    assert "agent_runtime_id=tau2_framwork_loop" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_unknown_runtime_id_even_when_it_contains_known_scheduler_name(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_unknown_framework_loop_runtime_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters[0]" in completed.stderr
    assert "agent_runtime_id=unknown_framework_loop_runtime" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_legacy_agent_that_would_be_skipped_when_another_agent_is_valid(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("legacy_agents_one_missing_scheduler.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agents[0]" in completed.stderr
    assert "missing_scheduler_agent" in completed.stderr
    assert "scheduler" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_backend_config_non_string_key(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_backend_config_numeric_key.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "backends[0].config" in completed.stderr
    assert "non-string key" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_unrefenced_static_backend_that_would_be_dropped(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_unreferenced_static_backend.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "unused_model" in completed.stderr
    assert "not referenced by agent_backends" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_static_backend_missing_type(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_static_backend_missing_type.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "backends[0].type" in completed.stderr
    assert "string" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_top_level_numeric_unsupported_key_without_traceback(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_top_level_numeric_unsupported_key.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "123" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "TypeError" not in completed.stderr
    assert not output_path.exists()


def test_cli_reports_nested_numeric_unsupported_key_without_traceback(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("legacy_agent_scheduler_numeric_extra_key.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agents[0].scheduler" in completed.stderr
    assert "123" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert "TypeError" not in completed.stderr
    assert not output_path.exists()


def test_cli_reports_top_level_scheduler_backend_id_ambiguous_with_agent_backend_id(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_top_level_scheduler_backend_id_ambiguous.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "scheduler.backend_id" in completed.stderr
    assert "agent_backend_id" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_loader_validation_failure_and_does_not_write_output(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"
    env = os.environ.copy()
    env.pop("TASK12A_REQUIRED_KEY", None)
    assert "TASK12A_REQUIRED_KEY" not in env

    completed = _run_migration_cli("v1_loader_validation_missing_env.yaml", output_path, env=env)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "loader validation failed" in completed.stderr
    assert "missing key" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_top_level_scheduler_backend_id_with_role_adapter_would_be_dropped(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_plus_top_level_scheduler_backend_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "scheduler.backend_id" in completed.stderr
    assert "role_adapters" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_role_adapters_mixed_with_top_level_agent_backend_id(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_plus_top_level_agent_backend_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters" in completed.stderr
    assert "top-level agent_backend_id" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_role_adapters_mixed_with_top_level_scheduler_model(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_plus_top_level_scheduler_model.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters" in completed.stderr
    assert "top-level scheduler" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_legacy_agent_scheduler_model_must_be_string(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("legacy_agent_scheduler_non_string_model.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agents[0].scheduler.model" in completed.stderr
    assert "string" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_role_adapter_adapter_id_must_be_string(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("v1_role_adapter_numeric_adapter_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters[0].adapter_id" in completed.stderr
    assert "string" in completed.stderr
    assert not output_path.exists()


def test_cli_reports_legacy_agent_agent_id_must_be_string(tmp_path: Path) -> None:
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli("legacy_agent_numeric_agent_id.yaml", output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "agents[0].agent_id" in completed.stderr
    assert "string" in completed.stderr
    assert not output_path.exists()


def test_cli_rejects_same_input_and_output_path_without_overwriting(tmp_path: Path) -> None:
    same_path = tmp_path / "same.yaml"
    original = (FIXTURE_DIR / "v1_role_adapter_minimal.yaml").read_text(encoding="utf-8")
    same_path.write_text(original, encoding="utf-8")

    completed = _run_migration_cli_paths(same_path, same_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "input and output paths must be different" in completed.stderr
    assert same_path.read_text(encoding="utf-8") == original


def test_cli_reports_invalid_yaml_without_traceback(tmp_path: Path) -> None:
    input_path = tmp_path / "invalid.yaml"
    input_path.write_text("metadata: [unterminated\n", encoding="utf-8")
    output_path = tmp_path / "migrated.yaml"

    completed = _run_migration_cli_paths(input_path, output_path)

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "failed to read input YAML" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output_path.exists()
