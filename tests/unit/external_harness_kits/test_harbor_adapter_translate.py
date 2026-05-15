from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
from typing import Any

import pytest

from gage_eval.external_harness_kits.errors import ExternalHarnessError
from gage_eval.external_harness_kits.harbor import launcher as harbor_launcher
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from gage_eval.role.role_manager import RoleManager
from gage_eval.role.adapters.harbor import HarborAdapter, HarborInvocation, _write_raw_input_artifacts
from gage_eval.external_harness_kits.base import TaskBatchHarnessRequest


def _task_dir(tmp_path: Path) -> Path:
    path = tmp_path / "gpt2-codegolf"
    (path / "environment").mkdir(parents=True, exist_ok=True)
    (path / "tests").mkdir(exist_ok=True)
    (path / "task.toml").write_text("[task]\n", encoding="utf-8")
    (path / "instruction.md").write_text("Write a tiny GPT-2 implementation.\n", encoding="utf-8")
    (path / "tests" / "test.sh").write_text("#!/usr/bin/env bash\nexit 0\n", encoding="utf-8")
    return path


def _base_payload(tmp_path: Path) -> dict[str, Any]:
    task_path = _task_dir(tmp_path)
    backend = {
        "backend_id": "lmstudio_local",
        "type": "litellm",
        "config": {
            "provider": "lm_studio",
            "custom_llm_provider": "lm_studio",
            "model": "lm_studio/qwen/qwen3.5-9b",
            "api_base": "http://127.0.0.1:1234/v1",
            "api_key": "EMPTY",
            "generation_parameters": {
                "temperature": 0.0,
                "max_new_tokens": 4096,
            },
            "model_info": {
                "max_input_tokens": 32768,
                "max_output_tokens": 4096,
                "input_cost_per_token": 0,
                "output_cost_per_token": 0,
            },
        },
    }
    dataset = {
        "dataset_id": "tb2_local_case",
        "loader": "harbor_local_path",
        "params": {
            "path": str(task_path),
            "task_name": "gpt2-codegolf",
            "path_kind": "task",
            "path_scope": "host",
        },
    }
    environment = {
        "env_id": "harbor_trial_docker",
        "provider": "docker",
        "resources": {
            "cpus": 2,
            "memory_mb": 4096,
        },
    }
    role_adapter = {
        "adapter_id": "harbor_tb2",
        "role_type": "external_harness",
        "backend_id": "lmstudio_local",
        "env_id": "harbor_trial_docker",
        "trial_policy": {
            "trials": 1,
            "aggregation": "all",
        },
        "params": {
            "harness": {
                "launcher": {"mode": "python_subprocess"},
                "agent": {
                    "kind": "base_agent",
                    "name": "terminus-2",
                },
                "job_options": {
                    "timeout_multiplier": 1.0,
                    "max_retries": 0,
                },
                "poll_interval_s": 5,
            }
        },
    }
    task = {
        "task_id": "terminal_bench_20_lmstudio_smoke",
        "dataset_id": "tb2_local_case",
        "execution_mode": "task_batch_harness",
        "max_samples": 1,
        "concurrency": 1,
    }
    return {
        "run_id": "run-001",
        "workdir": str(tmp_path / "gage-run" / "external_harness"),
        "task": task,
        "dataset": dataset,
        "datasets": [dataset],
        "backend": backend,
        "backends": [backend],
        "environment": environment,
        "environments": [environment],
        "role_adapter": role_adapter,
    }


def _translate(
    payload: dict[str, Any],
    *,
    registry_probe: Callable[..., bool] | None = None,
    installed_client_probe: Callable[..., bool] | None = None,
    local_path_visible_probe: Callable[..., bool] | None = None,
):
    adapter = HarborAdapter(
        adapter_id="harbor_tb2",
        registry_probe=registry_probe,
        installed_client_probe=installed_client_probe,
        local_path_visible_probe=local_path_visible_probe,
    )
    return adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))


@pytest.mark.fast
def test_harbor_adapter_exports_from_role_adapter_namespace() -> None:
    from gage_eval.role.adapters import HarborAdapter as ExportedHarborAdapter

    assert ExportedHarborAdapter is HarborAdapter


@pytest.mark.fast
def test_translates_lmstudio_base_agent_tb2_local_task_to_valid_job_config_without_taskconfig_source(
    tmp_path: Path,
) -> None:
    from harbor.models.job.config import JobConfig

    payload = _base_payload(tmp_path)

    plan = _translate(payload)

    job_config = plan.payload["job_config"]
    JobConfig.model_validate(job_config)
    assert job_config["job_name"] == "run_001__terminal_bench_20_lmstudio_smoke"
    assert job_config["n_attempts"] == 1
    assert job_config["n_concurrent_trials"] == 1
    assert job_config["timeout_multiplier"] == 1.0
    assert job_config["retry"]["max_retries"] == 0
    assert job_config["environment"] == {
        "type": "docker",
        "delete": False,
        "override_cpus": 2,
        "override_memory_mb": 4096,
    }
    assert job_config["datasets"] == []
    assert job_config["tasks"] == [{"path": payload["dataset"]["params"]["path"]}]
    assert "source" not in job_config["tasks"][0]
    agent = job_config["agents"][0]
    assert agent["name"] == "terminus-2"
    assert agent["model_name"] == "lm_studio/qwen/qwen3.5-9b"
    assert agent["kwargs"]["api_base"] == "http://127.0.0.1:1234/v1"
    assert agent["kwargs"]["custom_llm_provider"] == "lm_studio"
    assert agent["kwargs"]["temperature"] == 0.0
    assert agent["kwargs"]["llm_call_kwargs"]["max_tokens"] == 4096
    assert agent["kwargs"]["model_info"]["max_input_tokens"] == 32768
    assert plan.payload["adapter_projection"] == {"n_concurrent": 1}
    assert isinstance(plan.payload["invocation"], HarborInvocation)
    assert "EMPTY" not in str(job_config)
    assert all("EMPTY" not in part for part in plan.payload["invocation"].launcher_argv)


@pytest.mark.fast
def test_base_agent_receives_backend_provider_for_openai_compatible_local_models(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    backend_config = payload["backend"]["config"]
    backend_config.pop("custom_llm_provider")
    backend_config["provider"] = "openai"
    backend_config["model"] = "qwen/qwen3.5-9b"

    plan = _translate(payload)

    agent = plan.payload["job_config"]["agents"][0]
    assert agent["model_name"] == "qwen/qwen3.5-9b"
    assert agent["kwargs"]["api_base"] == "http://127.0.0.1:1234/v1"
    assert agent["kwargs"]["custom_llm_provider"] == "openai"


@pytest.mark.fast
def test_harbor_raw_input_artifacts_redact_backend_secrets(tmp_path: Path) -> None:
    invocation = HarborInvocation(
        job_name="job",
        jobs_dir=tmp_path / "jobs",
        job_config_path=tmp_path / "input.json",
        job_config={
            "agents": [
                {
                    "name": "terminus-2",
                    "kwargs": {
                        "api_key": "sk-testsecret123456",
                        "headers": {"Authorization": "Bearer abc123"},
                    },
                }
            ]
        },
        launcher_mode="python_subprocess",
        launcher_argv=["python", "-m", "harbor"],
        environ={"OPENAI_API_KEY": "sk-envsecret123456"},
        workdir=tmp_path / "work",
        expected_total_trials=1,
    )

    _write_raw_input_artifacts(invocation)

    serialized = "\n".join(
        [
            (invocation.workdir / "invocation.json").read_text(encoding="utf-8"),
            (invocation.workdir / "job_config.json").read_text(encoding="utf-8"),
        ]
    )
    assert "sk-testsecret123456" not in serialized
    assert "Bearer abc123" not in serialized
    assert "<redacted:" in serialized


@pytest.mark.fast
def test_launcher_invocation_preserves_repo_pythonpath_for_subprocess(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)

    plan = _translate(payload)

    pythonpath = plan.payload["invocation"].environ.get("PYTHONPATH", "")
    assert str(Path("src").resolve()) in pythonpath.split(os.pathsep)


@pytest.mark.fast
def test_translates_harbor_registry_to_flat_dataset_config(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["dataset"] = {
        "dataset_id": "tb2_registry",
        "loader": "harbor_registry",
        "params": {
            "ref": "terminal-bench@2.0",
            "registry_path": "/tmp/harbor-registry.json",
            "n_tasks": 1,
            "task_names": ["gpt2-codegolf"],
        },
    }
    payload["datasets"] = [payload["dataset"]]
    payload["task"]["dataset_id"] = "tb2_registry"

    plan = _translate(payload, registry_probe=lambda *_args, **_kwargs: True)

    assert plan.payload["job_config"]["datasets"] == [
        {
            "name": "terminal-bench",
            "version": "2.0",
            "registry_path": str(Path("/tmp/harbor-registry.json").resolve()),
            "task_names": ["gpt2-codegolf"],
            "n_tasks": 1,
        }
    ]
    assert plan.payload["job_config"]["tasks"] == []
    assert "registry" not in plan.payload["job_config"]["datasets"][0]


@pytest.mark.fast
def test_launch_stages_local_registry_mirror_with_absolute_task_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry_root = tmp_path / "registry-root"
    task_path = _task_dir(registry_root)
    registry_path = registry_root / "registry.json"
    registry_path.write_text(
        json.dumps(
            [
                {
                    "name": "terminal-bench",
                    "version": "2.0",
                    "description": "local mirror",
                    "tasks": [{"name": "gpt2-codegolf", "path": task_path.name}],
                    "metrics": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    payload = _base_payload(tmp_path)
    payload["dataset"] = {
        "dataset_id": "tb2_registry",
        "loader": "harbor_registry",
        "params": {
            "ref": "terminal-bench@2.0",
            "registry_path": str(registry_path),
            "n_tasks": 1,
            "task_names": ["gpt2-codegolf"],
        },
    }
    payload["datasets"] = [payload["dataset"]]
    payload["task"]["dataset_id"] = "tb2_registry"
    adapter = HarborAdapter(adapter_id="harbor_tb2", registry_probe=lambda *_args, **_kwargs: True)
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))
    captured: dict[str, Any] = {}

    def fake_launcher(**kwargs: Any) -> harbor_launcher.LauncherSubprocessResult:
        config_path = Path(kwargs["config_path"])
        captured["launcher_input"] = json.loads(config_path.read_text(encoding="utf-8"))
        result_file = Path(kwargs["result_file"])
        result_file.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        return harbor_launcher.LauncherSubprocessResult(
            argv=[],
            exit_code=0,
            timed_out=False,
            result_file=result_file,
            stdout_path=result_file.parent / "launcher.stdout.log",
            stderr_path=result_file.parent / "launcher.stderr.log",
        )

    monkeypatch.setattr(harbor_launcher, "run_launcher_subprocess", fake_launcher)

    adapter.launch(plan)

    staged_registry_path = Path(captured["launcher_input"]["job_config"]["datasets"][0]["registry_path"])
    staged_registry = json.loads(staged_registry_path.read_text(encoding="utf-8"))
    assert staged_registry_path.parent == Path(payload["workdir"]) / "registry_mirrors"
    assert staged_registry[0]["tasks"][0]["path"] == str(task_path.resolve())
    assert json.loads(registry_path.read_text(encoding="utf-8"))[0]["tasks"][0]["path"] == task_path.name


@pytest.mark.fast
def test_shutdown_marks_active_harbor_invocation_cancelled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _base_payload(tmp_path)
    adapter = HarborAdapter(adapter_id="harbor_tb2")
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))

    def fake_launcher(**kwargs: Any) -> harbor_launcher.LauncherSubprocessResult:
        adapter.shutdown()
        result_file = Path(kwargs["result_file"])
        result_file.parent.mkdir(parents=True, exist_ok=True)
        result_file.write_text(json.dumps({"exit_code": 0}), encoding="utf-8")
        return harbor_launcher.LauncherSubprocessResult(
            argv=[],
            exit_code=0,
            timed_out=False,
            result_file=result_file,
            stdout_path=result_file.parent / "launcher.stdout.log",
            stderr_path=result_file.parent / "launcher.stderr.log",
        )

    monkeypatch.setattr(harbor_launcher, "run_launcher_subprocess", fake_launcher)

    adapter.launch(plan)

    invocation = plan.payload["invocation"]
    workdir_marker = invocation.workdir / "cancelled.json"
    job_marker = invocation.jobs_dir / invocation.job_name / "cancelled.json"
    workdir_payload = json.loads(workdir_marker.read_text(encoding="utf-8"))
    job_payload = json.loads(job_marker.read_text(encoding="utf-8"))
    assert workdir_payload["status"] == "cancelled"
    assert workdir_payload["reason"] == "adapter_shutdown"
    assert workdir_payload["job_name"] == invocation.job_name
    assert job_payload == workdir_payload


@pytest.mark.fast
def test_launch_removes_active_invocation_when_launcher_raises(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _base_payload(tmp_path)
    adapter = HarborAdapter(adapter_id="harbor_tb2")
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))

    def fake_launcher(**_kwargs: Any) -> harbor_launcher.LauncherSubprocessResult:
        raise RuntimeError("launcher crashed")

    monkeypatch.setattr(harbor_launcher, "run_launcher_subprocess", fake_launcher)

    with pytest.raises(RuntimeError, match="launcher crashed"):
        adapter.launch(plan)

    assert adapter._active_invocations == {}


@pytest.mark.fast
def test_role_manager_shutdown_invokes_harbor_adapter_shutdown_marker(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    adapter = HarborAdapter(adapter_id="harbor_tb2")
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))
    invocation = plan.payload["invocation"]
    invocation.workdir.mkdir(parents=True, exist_ok=True)
    invocation.jobs_dir.mkdir(parents=True, exist_ok=True)
    adapter._active_invocations[invocation.job_name] = invocation
    manager = RoleManager(ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)]))
    manager._adapters = {adapter.adapter_id: adapter}
    manager._role_pools = {}

    manager.shutdown()

    assert (invocation.workdir / "cancelled.json").is_file()
    assert json.loads((invocation.workdir / "cancelled.json").read_text(encoding="utf-8"))["reason"] == "adapter_shutdown"


@pytest.mark.fast
def test_shutdown_marks_all_active_harbor_invocations_cancelled(tmp_path: Path) -> None:
    adapter = HarborAdapter(adapter_id="harbor_tb2")
    invocations = []
    for index in range(2):
        payload = _base_payload(tmp_path / f"case-{index}")
        payload["run_id"] = f"run-{index}"
        payload["workdir"] = str(tmp_path / f"run-{index}" / "external_harness")
        plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))
        invocation = plan.payload["invocation"]
        invocation.workdir.mkdir(parents=True, exist_ok=True)
        invocation.jobs_dir.mkdir(parents=True, exist_ok=True)
        adapter._active_invocations[invocation.job_name] = invocation
        invocations.append(invocation)

    adapter.shutdown()

    assert len(invocations) == 2
    for invocation in invocations:
        marker = json.loads((invocation.workdir / "cancelled.json").read_text(encoding="utf-8"))
        assert marker["status"] == "cancelled"
        assert marker["job_name"] == invocation.job_name


@pytest.mark.fast
def test_raises_invalid_model_when_lmstudio_model_omits_litellm_provider_prefix(
    tmp_path: Path,
) -> None:
    payload = _base_payload(tmp_path)
    payload["backend"]["config"]["model"] = "qwen/qwen3.5-9b"

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.translate.invalid_model"
    assert "lm_studio/qwen/qwen3.5-9b" in str(exc_info.value)
    assert "real model id" in str(exc_info.value)


@pytest.mark.fast
def test_deep_merges_model_info_and_raises_model_info_conflict(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["role_adapter"]["params"]["harness"]["agent"]["kwargs"] = {
        "model_info": {
            "max_output_tokens": 4096,
            "supports_reasoning": False,
        }
    }

    job_config = _translate(payload).payload["job_config"]

    assert job_config["agents"][0]["kwargs"]["model_info"]["supports_reasoning"] is False

    payload["role_adapter"]["params"]["harness"]["agent"]["kwargs"]["model_info"][
        "max_output_tokens"
    ] = 8192
    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.translate.model_info_conflict"


@pytest.mark.fast
def test_installed_client_shape_differs_from_base_agent_and_requires_preflight(
    tmp_path: Path,
) -> None:
    base_payload = _base_payload(tmp_path)
    installed_payload = _base_payload(tmp_path)
    agent = installed_payload["role_adapter"]["params"]["harness"]["agent"]
    agent.update(
        {
            "kind": "installed_client",
            "name": "claude-code",
            "extra_env": {"OPENAI_BASE_URL": "http://host.docker.internal:1234/v1"},
        }
    )

    base_agent = _translate(base_payload).payload["job_config"]["agents"][0]
    installed_agent = _translate(
        installed_payload,
        installed_client_probe=lambda *_args, **_kwargs: True,
    ).payload["job_config"]["agents"][0]

    assert base_agent["kwargs"]["api_base"] == "http://127.0.0.1:1234/v1"
    assert "env" not in base_agent
    assert "api_base" not in installed_agent.get("kwargs", {})
    assert installed_agent["env"]["OPENAI_BASE_URL"] == "http://host.docker.internal:1234/v1"

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(installed_payload, installed_client_probe=lambda *_args, **_kwargs: False)

    assert exc_info.value.code == "external_harness.translate.installed_client_incompatible"


@pytest.mark.fast
@pytest.mark.parametrize(
    ("method_name", "expected_code"),
    [
        ("_preflight_harbor_import", "external_harness.runtime.harbor_unavailable"),
        ("_preflight_harbor_job_api", "external_harness.runtime.harbor_api_incompatible"),
        ("_preflight_registry_ref", "external_harness.runtime.registry_not_found"),
        ("_preflight_job_config_io", "external_harness.environment.io_unusable"),
        ("_preflight_provider_compatibility", "external_harness.environment.provider_mismatch"),
        ("_preflight_model_endpoint", "external_harness.environment.model_endpoint_unreachable"),
        ("_preflight_docker_trial", "external_harness.environment.docker_unavailable"),
        ("_preflight_e2b_trial", "external_harness.environment.e2b_unavailable"),
        ("_preflight_jobs_dir_artifact_pull", "external_harness.environment.artifact_pull_failed"),
    ],
)
def test_initialize_raises_appendix_b_error_codes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    expected_code: str,
) -> None:
    payload = _base_payload(tmp_path)
    adapter = HarborAdapter(
        adapter_id="harbor_tb2",
        model_endpoint_probe=lambda **_kwargs: True,
        docker_probe=lambda: True,
        artifact_pull_probe=lambda **_kwargs: True,
    )
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))

    def fail(_context: dict[str, Any]) -> None:
        raise RuntimeError("probe failed")

    monkeypatch.setattr(adapter, method_name, fail)

    with pytest.raises(ExternalHarnessError) as exc_info:
        adapter._initialize(plan)

    assert exc_info.value.code == expected_code


@pytest.mark.fast
def test_initialize_without_plan_does_not_run_plan_bound_preflights(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.job import Job

    adapter = HarborAdapter(
        adapter_id="harbor_tb2",
        artifact_pull_probe=lambda **_kwargs: False,
    )

    def fail_if_called(_context: dict[str, Any]) -> None:
        raise AssertionError("plan-bound preflight should not run before translation")

    monkeypatch.setattr(adapter, "_preflight_registry_ref", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_job_config_io", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_provider_compatibility", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_model_endpoint", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_docker_trial", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_e2b_trial", fail_if_called)
    monkeypatch.setattr(adapter, "_preflight_jobs_dir_artifact_pull", fail_if_called)

    async def fail_create(cls, config):
        del cls, config
        raise AssertionError("initialize without plan must not create a Harbor job")

    monkeypatch.setattr(Job, "create", classmethod(fail_create))

    adapter.initialize()


@pytest.mark.fast
def test_plan_bound_preflight_job_api_uses_run_workdir_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from harbor.job import Job

    payload = _base_payload(tmp_path)
    adapter = HarborAdapter(
        adapter_id="harbor_tb2",
        model_endpoint_probe=lambda **_kwargs: True,
        docker_probe=lambda: True,
        artifact_pull_probe=lambda **_kwargs: True,
    )
    plan = adapter.translate(TaskBatchHarnessRequest(adapter_id="harbor_tb2", payload=payload))
    invocation = plan.payload["invocation"]
    captured: dict[str, Path] = {}

    async def fake_create(cls, config):
        del cls
        jobs_dir = Path(str(config.jobs_dir))
        captured["jobs_dir"] = jobs_dir
        (jobs_dir / config.job_name).mkdir(parents=True, exist_ok=True)
        (jobs_dir / config.job_name / "job.log").write_text("preflight\n", encoding="utf-8")
        return object()

    monkeypatch.setattr(Job, "create", classmethod(fake_create))

    adapter._initialize(plan)

    expected_jobs_dir = invocation.workdir / "_preflight" / "jobs"
    assert captured["jobs_dir"] == expected_jobs_dir
    assert not expected_jobs_dir.exists()
    assert not (Path.cwd() / "jobs" / "gage_preflight").exists()


@pytest.mark.fast
@pytest.mark.parametrize(
    ("case_name", "mutate", "expected_code"),
    [
        (
            "backend_missing",
            lambda payload: (payload.update({"backend": None, "backends": []})),
            "external_harness.translate.backend_missing",
        ),
        (
            "invalid_model",
            lambda payload: payload["backend"]["config"].update({"model": ""}),
            "external_harness.translate.invalid_model",
        ),
        (
            "unresolved_secret",
            lambda payload: payload["backend"]["config"].update({"api_key": "${MISSING_SECRET}"}),
            "external_harness.translate.unresolved_secret",
        ),
        (
            "invalid_ref",
            lambda payload: (
                payload["dataset"].update({"loader": "harbor_registry"}),
                payload["dataset"]["params"].clear(),
                payload["dataset"]["params"].update({"ref": "terminal-bench"}),
            ),
            "external_harness.translate.invalid_ref",
        ),
        (
            "registry_not_found",
            lambda payload: (
                payload["dataset"].update({"loader": "harbor_registry"}),
                payload["dataset"]["params"].clear(),
                payload["dataset"]["params"].update({"ref": "terminal-bench@2.0"}),
            ),
            "external_harness.runtime.registry_not_found",
        ),
        (
            "local_path_not_found",
            lambda payload: payload["dataset"]["params"].update({"path": "/tmp/gage-missing-harbor-task"}),
            "external_harness.runtime.local_path_not_found",
        ),
        (
            "local_path_not_visible",
            lambda payload: payload,
            "external_harness.runtime.local_path_not_visible",
        ),
        (
            "invalid_dataset_params",
            lambda payload: payload["dataset"]["params"].update({"path_kind": "dataset"}),
            "external_harness.config.invalid_dataset_params",
        ),
        (
            "invalid_agent",
            lambda payload: payload["role_adapter"]["params"]["harness"]["agent"].update(
                {"name": "not-a-harbor-agent"}
            ),
            "external_harness.translate.invalid_agent",
        ),
        (
            "backend_agent_bridge_failed",
            lambda payload: payload["backend"]["config"].pop("api_base"),
            "external_harness.translate.backend_agent_bridge_failed",
        ),
        (
            "model_info_required",
            lambda payload: (
                payload["backend"]["config"].update({"model": "hosted_vllm/qwen"}),
                payload["backend"]["config"].pop("model_info"),
            ),
            "external_harness.translate.model_info_required",
        ),
        (
            "model_info_conflict",
            lambda payload: payload["role_adapter"]["params"]["harness"]["agent"].update(
                {"kwargs": {"model_info": {"max_input_tokens": 1}}}
            ),
            "external_harness.translate.model_info_conflict",
        ),
        (
                "installed_client_incompatible",
                lambda payload: payload["role_adapter"]["params"]["harness"]["agent"].update(
                    {
                        "kind": "installed_client",
                        "name": "terminus-2",
                        "extra_env": {"OPENAI_BASE_URL": "http://host.docker.internal:1234/v1"},
                    }
                ),
            "external_harness.translate.installed_client_incompatible",
        ),
        (
            "invalid_trials",
            lambda payload: payload["role_adapter"].update({"trial_policy": {"trials": 0}}),
            "external_harness.translate.invalid_trials",
        ),
        (
            "environment_bridge_failed",
            lambda payload: payload.update({"environment": None, "environments": []}),
            "external_harness.translate.environment_bridge_failed",
        ),
        (
            "secret_serialization_blocked",
            lambda payload: payload["role_adapter"]["params"]["harness"]["agent"].update(
                {"kwargs": {"api_key": "sk-test-secret"}}
            ),
            "external_harness.translate.secret_serialization_blocked",
        ),
    ],
)
def test_translate_raises_appendix_c_error_codes(
    tmp_path: Path,
    case_name: str,
    mutate: Callable[[dict[str, Any]], Any],
    expected_code: str,
) -> None:
    payload = _base_payload(tmp_path)
    mutate(payload)
    registry_probe = None
    local_path_visible_probe = None
    if case_name == "registry_not_found":
        registry_probe = lambda *_args, **_kwargs: False
    if case_name == "local_path_not_visible":
        local_path_visible_probe = lambda *_args, **_kwargs: False

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(
            payload,
            registry_probe=registry_probe,
            local_path_visible_probe=local_path_visible_probe,
        )

    assert exc_info.value.code == expected_code


@pytest.mark.fast
def test_raises_invalid_concurrency_when_task_concurrency_is_less_than_one(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["task"]["concurrency"] = 0

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_concurrency"


@pytest.mark.fast
def test_raises_invalid_concurrency_for_malformed_concurrency(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["task"]["concurrency"] = "many"

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_concurrency"


@pytest.mark.fast
def test_raises_invalid_concurrency_for_non_finite_concurrency(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["task"]["concurrency"] = float("inf")

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_concurrency"


@pytest.mark.fast
def test_raises_invalid_concurrency_for_fractional_concurrency(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["task"]["concurrency"] = 1.9

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_concurrency"


@pytest.mark.fast
@pytest.mark.parametrize("timeout_multiplier", [float("nan"), float("inf")])
def test_raises_invalid_dataset_params_for_non_finite_timeout_multiplier(
    tmp_path: Path,
    timeout_multiplier: float,
) -> None:
    payload = _base_payload(tmp_path)
    payload["role_adapter"]["trial_policy"]["timeout_multiplier"] = timeout_multiplier

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_dataset_params"


@pytest.mark.fast
def test_raises_invalid_loader_for_non_harbor_dataset_loader(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["dataset"]["loader"] = "jsonl"

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.invalid_loader"


@pytest.mark.fast
def test_raises_secret_agent_env_forbidden_for_extra_env_secret(tmp_path: Path) -> None:
    payload = _base_payload(tmp_path)
    payload["role_adapter"]["params"]["harness"]["agent"]["extra_env"] = {
        "OPENAI_API_KEY": "sk-test-secret"
    }

    with pytest.raises(ExternalHarnessError) as exc_info:
        _translate(payload)

    assert exc_info.value.code == "external_harness.config.secret_agent_env_forbidden"


@pytest.mark.fast
def test_dry_run_returns_job_config_launcher_argv_and_redacted_invocation_without_launching(
    tmp_path: Path,
) -> None:
    payload = _base_payload(tmp_path)
    payload["dry_run"] = True
    payload["backend"]["config"]["api_key"] = "sk-live-value"

    plan = _translate(payload)

    dry_run = plan.payload["dry_run"]
    assert dry_run["job_config"] == plan.payload["job_config"]
    assert dry_run["launcher_argv"] == plan.payload["invocation"].launcher_argv
    assert dry_run["invocation"]["environ"]["OPENAI_API_KEY"]["is_secret"] is True
    assert dry_run["invocation"]["environ"]["OPENAI_API_KEY"]["value"] == "<redacted>"
    assert dry_run["invocation"]["environ"]["OPENAI_API_KEY"]["value_sha256"].startswith("sha256:")
    assert "sk-live-value" not in str(dry_run)
