from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

from gage_eval.agent_runtime.trace_schema import SampleRecord
from gage_eval.config.agentkit_v2 import (
    load_agentkit_v2_config_payload,
    materialize_agentkit_v2_runtime_config_payload,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_MIGRATION_SCRIPT = REPO_ROOT / "scripts" / "migrate_agentkit_v1_config_to_v2.py"
RUN_MIGRATION_SCRIPT = REPO_ROOT / "scripts" / "migrate_runs_v1_to_v2.py"


def test_migrates_tau2_legacy_smoke_yaml_to_agent_eval_config(tmp_path: Path) -> None:
    module = _load_module(CONFIG_MIGRATION_SCRIPT, "migrate_agentkit_v1_config_to_v2_task12b")
    input_path = _write_legacy_tau2_config(tmp_path)
    output_path = tmp_path / "tau2_v2.yaml"

    result = module.migrate_file(
        input_path,
        output_path,
    )

    assert result.ok, result.manual_fixes
    migrated = _read_yaml(output_path)
    assert _agent_eval_sections(migrated)
    assert "datasets" not in migrated
    assert "role_adapters" not in migrated
    assert migrated["benchmarks"] == [
        {
            "benchmark_id": "tau2_telecom_runtime",
            "kit_id": "tau2",
            "config": {
                "domain": "telecom",
                "user_simulator": {
                    "model": "${TAU2_USER_MODEL:-gpt-4.1}",
                    "model_args": {"temperature": "${TAU2_USER_TEMPERATURE:-0.0}"},
                },
            },
        }
    ]
    assert migrated["environments"][0]["provider"] == "local_process"
    assert migrated["environments"][0]["profile_id"] == "tau2_local"
    assert "user_model" not in migrated["environments"][0]["profile"]
    assert "runtime_configs" not in migrated["environments"][0]["profile"]
    assert "user_model" not in migrated["environments"][0].get("provider_config", {})
    assert migrated["dut_agents"][0]["trial_policy"] == {"trials": "${TAU2_NUM_TRIALS:-1}"}

    loaded = load_agentkit_v2_config_payload(output_path)
    assert loaded["benchmarks"][0]["kit_id"] == "tau2"
    assert loaded["dut_agents"][0]["trial_policy"] == {"trials": "<redacted:reference:TAU2_NUM_TRIALS>"}
    runtime_payload = materialize_agentkit_v2_runtime_config_payload(migrated, output_path)
    assert runtime_payload["dut_agents"][0]["trial_policy"] == {"trials": 1}


def test_migrates_swebench_legacy_smoke_yaml_to_agent_eval_config(tmp_path: Path) -> None:
    module = _load_module(CONFIG_MIGRATION_SCRIPT, "migrate_agentkit_v1_config_to_v2_task12b_swe")
    input_path = _write_legacy_swebench_config(tmp_path)
    output_path = tmp_path / "swebench_v2.yaml"

    result = module.migrate_file(
        input_path,
        output_path,
    )

    assert result.ok, result.manual_fixes
    migrated = _read_yaml(output_path)
    assert _agent_eval_sections(migrated)
    assert migrated["benchmarks"] == [
        {
            "benchmark_id": "swebench_pro_smoke_runtime_eval",
            "kit_id": "swebench",
            "config": {"split": "test"},
        }
    ]
    assert migrated["environments"][0]["provider"] == "docker"
    assert migrated["environments"][0]["resources"] == {"cpu": 4, "memory_gb": 8.0}
    assert "runtime_configs" not in migrated["environments"][0]["profile"]
    assert "provider_config" in migrated["environments"][0]
    assert migrated["agents"][0]["scheduler"] == {
        "type": "framework_loop",
        "backend_id": "gpt52_openai_http",
        "config": {"max_turns": 40},
    }
    migration = migrated["metadata"]["migration"]
    assert migration["legacy_judge_adapters"][0]["implementation"] == "swebench_docker"
    assert "datasets" in migration["preserved_legacy_sections"]
    assert "tasks" in migration["preserved_legacy_sections"]
    assert "role_adapters" in migration["preserved_legacy_sections"]
    assert "tasks[0].max_samples" in migration["runtime_unapplied_fields"]
    assert "datasets[0].loader" in migration["runtime_unapplied_fields"]

    loaded = load_agentkit_v2_config_payload(output_path)
    assert loaded["benchmarks"][0]["config"] == {"split": "test"}
    assert loaded["environments"][0]["provider"] == "docker"


def test_migrate_run_v1_to_trial_0001_artifact_layout_and_sample_record_schema(
    tmp_path: Path,
) -> None:
    module = _load_module(RUN_MIGRATION_SCRIPT, "migrate_runs_v1_to_v2_task12b")
    legacy_run = _write_legacy_run_fixture(tmp_path)
    output_base = tmp_path / "new-runs"

    result = module.migrate_run(legacy_run, output_base_dir=output_base)

    assert result.ok, result.manual_fixes
    output_run = output_base / "legacy-run-1"
    trial_root = output_run / "artifacts/tau2_telecom_runtime/sample-1/trials/trial_0001"
    assert (trial_root / "infra/trace.jsonl").exists()
    assert (trial_root / "infra/trial_result.json").exists()
    assert (trial_root / "agent/scheduler_result.json").exists()
    assert (trial_root / "verifier/verifier_result.json").exists()
    sample_record_path = output_run / "artifacts/tau2_telecom_runtime/sample-1/infra/sample_record.json"

    sample_record = SampleRecord.model_validate(json.loads(sample_record_path.read_text(encoding="utf-8")))
    assert sample_record.run_id == "legacy-run-1"
    assert sample_record.task_id == "tau2_telecom_runtime"
    assert sample_record.sample_id == "sample-1"
    assert sample_record.trial_results[0].trial_id == "trial_0001"
    assert sample_record.verifier_result["reward"] == 1.0

    samples_jsonl = output_run / "samples.jsonl"
    [line] = [json.loads(line) for line in samples_jsonl.read_text(encoding="utf-8").splitlines()]
    assert line["primary_trial_id"] == "trial_0001"
    assert line["reward"] == 1.0
    assert line["sample_record_ref"]["path"] == "artifacts/tau2_telecom_runtime/sample-1/infra/sample_record.json"


def test_run_migration_reports_manual_fix_when_sample_identity_is_unknown(tmp_path: Path) -> None:
    legacy_run = tmp_path / "legacy-run-missing-id"
    legacy_run.mkdir()
    (legacy_run / "samples.jsonl").write_text(json.dumps({"task_id": "task-1"}) + "\n", encoding="utf-8")
    output_base = tmp_path / "new-runs"

    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_MIGRATION_SCRIPT),
            "--input-run",
            str(legacy_run),
            "--output-base",
            str(output_base),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "sample_id" in completed.stderr
    assert not (output_base / "legacy-run-missing-id").exists()


def test_run_migration_reports_manual_fix_for_sanitized_sample_collision(tmp_path: Path) -> None:
    module = _load_module(RUN_MIGRATION_SCRIPT, "migrate_runs_v1_to_v2_task12b_collision")
    legacy_run = tmp_path / "legacy-run-collision"
    legacy_run.mkdir()
    records = [
        {
            "sample_id": "a/b",
            "task_id": "task-1",
            "model_output": {"answer": "a"},
            "judge_output": {"score": 0.0},
        },
        {
            "sample_id": "a_b",
            "task_id": "task-1",
            "model_output": {"answer": "b"},
            "judge_output": {"score": 1.0},
        },
    ]
    (legacy_run / "samples.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )

    result = module.migrate_run(legacy_run, output_base_dir=tmp_path / "new-runs")

    assert not result.ok
    assert any("sanitized sample/task id collision" in message for message in result.manual_fixes)
    assert not (tmp_path / "new-runs" / "legacy-run-collision").exists()


def test_run_migration_reports_manual_fix_when_result_fields_are_unknown(tmp_path: Path) -> None:
    module = _load_module(RUN_MIGRATION_SCRIPT, "migrate_runs_v1_to_v2_task12b_missing_results")
    legacy_run = tmp_path / "legacy-run-missing-results"
    legacy_run.mkdir()
    (legacy_run / "samples.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "task_id": "task-1"}) + "\n",
        encoding="utf-8",
    )

    result = module.migrate_run(legacy_run, output_base_dir=tmp_path / "new-runs")

    assert not result.ok
    assert any("missing agent output" in message for message in result.manual_fixes)
    assert any("missing verifier output" in message for message in result.manual_fixes)
    assert not (tmp_path / "new-runs" / "legacy-run-missing-results").exists()


def test_run_migration_cli_reports_write_failure_without_traceback_or_partial_output(
    tmp_path: Path,
) -> None:
    legacy_run = _write_legacy_run_fixture(tmp_path)
    output_base = tmp_path / "not-a-directory"
    output_base.write_text("blocks directory creation", encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            str(RUN_MIGRATION_SCRIPT),
            "--input-run",
            str(legacy_run),
            "--output-base",
            str(output_base),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not (output_base / "legacy-run-1").exists()


def test_config_migration_reports_manual_fix_for_unknown_agent_runtime(tmp_path: Path) -> None:
    input_path = tmp_path / "unknown-runtime.yaml"
    output_path = tmp_path / "migrated.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "api_version": "gage/v1",
                "kind": "PipelineConfig",
                "metadata": {"name": "unknown-runtime"},
                "datasets": [{"dataset_id": "tau2", "params": {"domain": "telecom"}}],
                "backends": [{"backend_id": "model", "type": "openai_http", "config": {"model": "gpt"}}],
                "sandbox_profiles": [{"sandbox_id": "tau2_local", "runtime": "tau2", "runtime_configs": {}}],
                "role_adapters": [
                    {
                        "adapter_id": "agent",
                        "role_type": "dut_agent",
                        "backend_id": "model",
                        "agent_runtime_id": "custom_agent_runtime",
                        "sandbox": {"sandbox_id": "tau2_local"},
                    }
                ],
                "tasks": [
                    {
                        "task_id": "tau2_task",
                        "dataset_id": "tau2",
                        "steps": [{"step": "inference", "adapter_id": "agent"}],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CONFIG_MIGRATION_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "custom_agent_runtime" in completed.stderr
    assert not output_path.exists()


def test_config_migration_reports_manual_fix_for_unknown_nested_pipeline_field(
    tmp_path: Path,
) -> None:
    input_path = tmp_path / "unknown-nested.yaml"
    output_path = tmp_path / "migrated.yaml"
    input_path.write_text(
        yaml.safe_dump(
            {
                "api_version": "gage/v1",
                "kind": "PipelineConfig",
                "metadata": {"name": "unknown-nested"},
                "datasets": [{"dataset_id": "tau2", "loader": "tau2_tasks", "params": {"domain": "telecom"}}],
                "backends": [{"backend_id": "model", "type": "openai_http", "config": {"model": "gpt"}}],
                "sandbox_profiles": [{"sandbox_id": "tau2_local", "runtime": "tau2", "runtime_configs": {}}],
                "role_adapters": [
                    {
                        "adapter_id": "agent",
                        "role_type": "dut_agent",
                        "backend_id": "model",
                        "agent_runtime_id": "tau2_framework_loop",
                        "sandbox": {"sandbox_id": "tau2_local"},
                        "mcp_client_id": "legacy-mcp",
                    }
                ],
                "tasks": [
                    {
                        "task_id": "tau2_task",
                        "dataset_id": "tau2",
                        "steps": [{"step": "inference", "adapter_id": "agent"}],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(CONFIG_MIGRATION_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "manual migration required" in completed.stderr
    assert "role_adapters[0] contains unsupported fields mcp_client_id" in completed.stderr
    assert not output_path.exists()


def test_config_migration_cli_reports_write_failure_without_partial_output(
    tmp_path: Path,
) -> None:
    input_path = _write_legacy_tau2_config(tmp_path)
    output_parent = tmp_path / "blocked-output-parent"
    output_parent.write_text("not a directory", encoding="utf-8")
    output_path = output_parent / "migrated.yaml"

    completed = subprocess.run(
        [
            sys.executable,
            str(CONFIG_MIGRATION_SCRIPT),
            "--input",
            str(input_path),
            "--output",
            str(output_path),
        ],
        check=False,
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 1
    assert "manual migration required: failed to write output YAML" in completed.stderr
    assert "Traceback" not in completed.stderr
    assert not output_path.exists()


def _load_module(path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _agent_eval_sections(payload: dict[str, Any]) -> bool:
    return {"backends", "agents", "benchmarks", "environments", "dut_agents"}.issubset(payload)


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    assert isinstance(payload, dict)
    return payload


def _write_legacy_run_fixture(tmp_path: Path) -> Path:
    legacy_run = tmp_path / "legacy-run-1"
    legacy_run.mkdir()
    sample_record = {
        "sample_id": "sample-1",
        "task_id": "tau2_telecom_runtime",
        "namespace": "tau2_telecom_runtime",
        "sample": {"metadata": {"domain": "telecom"}},
        "model_output": {"answer": "Done", "usage": {"total_tokens": 12}},
        "judge_output": {"status": "completed", "reward": 1.0, "resolved": True},
        "metrics": {"tau2_reward": {"value": 1.0}},
    }
    (legacy_run / "samples.jsonl").write_text(json.dumps(sample_record) + "\n", encoding="utf-8")
    (legacy_run / "summary.json").write_text(json.dumps({"sample_count": 1}), encoding="utf-8")
    return legacy_run


def _write_legacy_tau2_config(tmp_path: Path) -> Path:
    path = tmp_path / "legacy_tau2_telecom_runtime.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api_version": "gage/v1",
                "kind": "PipelineConfig",
                "metadata": {"name": "tau2_telecom_runtime"},
                "datasets": [
                    {
                        "dataset_id": "tau2_telecom_base",
                        "loader": "tau2_tasks",
                        "params": {
                            "domain": "telecom",
                            "task_split": "base",
                            "num_trials": "${TAU2_NUM_TRIALS:-1}",
                        },
                    }
                ],
                "backends": [
                    {
                        "backend_id": "tau2_openai_http",
                        "type": "openai_http",
                        "config": {
                            "base_url": "${TAU2_OPENAI_BASE_URL:-https://api.openai.com/v1}",
                            "model": "${TAU2_AGENT_MODEL:-gpt-4.1}",
                        },
                    }
                ],
                "sandbox_profiles": [
                    {
                        "sandbox_id": "tau2_local",
                        "runtime": "tau2",
                        "runtime_configs": {
                            "user_model": "${TAU2_USER_MODEL:-gpt-4.1}",
                            "user_model_args": {"temperature": "${TAU2_USER_TEMPERATURE:-0.0}"},
                        },
                    }
                ],
                "role_adapters": [
                    {
                        "adapter_id": "tau2_agent",
                        "role_type": "dut_agent",
                        "backend_id": "tau2_openai_http",
                        "agent_runtime_id": "tau2_framework_loop",
                        "sandbox": {"sandbox_id": "tau2_local"},
                        "params": {"max_turns": 200},
                    }
                ],
                "tasks": [
                    {
                        "task_id": "tau2_telecom_runtime",
                        "dataset_id": "tau2_telecom_base",
                        "steps": [{"step": "inference", "adapter_id": "tau2_agent"}],
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path


def _write_legacy_swebench_config(tmp_path: Path) -> Path:
    path = tmp_path / "legacy_swebench_pro_smoke_runtime.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "api_version": "gage/v1alpha1",
                "kind": "PipelineConfig",
                "metadata": {"name": "swebench_pro_smoke_runtime"},
                "datasets": [
                    {
                        "dataset_id": "swebench_pro_smoke",
                        "loader": "hf_hub",
                        "hub_params": {"split": "test"},
                        "params": {
                            "preprocess": "swebench_pro_standardizer",
                            "preprocess_kwargs": {
                                "smoke_ids_path": "third_party/swebench_pro/run_scripts/smoke_instance_ids.txt"
                            },
                        },
                    }
                ],
                "backends": [
                    {
                        "backend_id": "gpt52_openai_http",
                        "type": "openai_http",
                        "config": {"base_url": "https://api.openai.com/v1", "model": "gpt-5.2"},
                    }
                ],
                "sandbox_profiles": [
                    {
                        "sandbox_id": "swebench_runtime",
                        "runtime": "docker",
                        "resources": {"cpu": 4, "memory": "8g"},
                        "runtime_configs": {
                            "network_mode": "none",
                            "platform": "linux/amd64",
                            "entrypoint": "/bin/bash",
                            "command": ["-c", "sleep 3600"],
                            "exec_workdir": "/app",
                        },
                    }
                ],
                "role_adapters": [
                    {
                        "adapter_id": "swebench_dut_agent",
                        "role_type": "dut_agent",
                        "backend_id": "gpt52_openai_http",
                        "agent_runtime_id": "swebench_framework_loop",
                        "sandbox": {"sandbox_id": "swebench_runtime"},
                        "params": {"max_turns": 40},
                    },
                    {
                        "adapter_id": "swebench_runtime_judge",
                        "role_type": "judge_extend",
                        "sandbox": {"sandbox_id": "swebench_runtime"},
                        "params": {
                            "implementation": "swebench_docker",
                            "implementation_params": {
                                "scripts_dir": "third_party/swebench_pro/run_scripts",
                                "dockerfiles_dir": "third_party/swebench_pro/dockerfiles",
                            },
                        },
                    },
                ],
                "tasks": [
                    {
                        "task_id": "swebench_pro_smoke_runtime_eval",
                        "dataset_id": "swebench_pro_smoke",
                        "steps": [
                            {"step": "inference", "adapter_id": "swebench_dut_agent"},
                            {"step": "judge", "adapter_id": "swebench_runtime_judge"},
                            {"step": "auto_eval"},
                        ],
                        "max_samples": 11,
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return path
