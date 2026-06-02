from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
import yaml

from gage_eval.config.loader import load_pipeline_config_payload
from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.config.registry import ConfigRegistry
from gage_eval.evaluation.runtime_builder import build_runtime
from gage_eval.external_harness_kits.errors import ExternalHarnessError
from gage_eval.role.adapters import harbor as harbor_module
from gage_eval.role.adapters.harbor import HarborAdapter
from gage_eval.external_harness_kits.base import TaskBatchHarnessRequest
from gage_eval.role.resource_profile import NodeResource, ResourceProfile
from tests._support.external_harness_kits import fake_harbor_launcher


FIXTURE_DIR = Path("tests/fixtures/external_harness_kits")
LMSTUDIO_FIXTURE = FIXTURE_DIR / "terminal_bench_1case_lmstudio.yaml"
INSTALLED_CLIENT_FIXTURE = FIXTURE_DIR / "terminal_bench_1case_installed_client_mock.yaml"
SWEBENCH_PRO_CONFIG = Path("config/custom/external_harness_kits/harbor_swebench_pro_lmstudio_1case.yaml")
SWEBENCH_PRO_ANSIBLE_TASK = (
    "instance_ansible__ansible-11c1777d56664b1acb56b387a1ad6aeadef1391d"
    "-v0f01c69f1e2528b935359cfe578530722bca2c59"
)


def test_mocked_harbor_task_batch_pipeline_runs_and_reports_one_sample(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _load_payload(LMSTUDIO_FIXTURE)
    config = PipelineConfig.from_dict(payload)
    run_id = "task15-mocked-harbor"
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setattr(
        harbor_module.harbor_launcher,
        "run_launcher_subprocess",
        fake_harbor_launcher.run_fake_launcher,
    )
    monkeypatch.setattr(
        harbor_module.HarborAdapter,
        "_initialize",
        lambda self, plan=None: None,
    )

    runtime = build_runtime(
        config,
        ConfigRegistry(),
        _resource_profile(),
        trace=_trace(run_id),
    )

    runtime.run()

    run_dir = tmp_path / run_id
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    samples = [
        json.loads(line)
        for line in (run_dir / "samples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    sample_id = samples[0]["sample_id"]
    artifact_root = run_dir / "artifacts" / "tb2_one_case" / sample_id
    raw_root = run_dir / "external_harness"
    provider_root = raw_root / "tb2_one_case" / "harbor_tb2"
    raw_job_tree = provider_root / "jobs"
    raw_manifest = json.loads((raw_root / "manifest.json").read_text(encoding="utf-8"))
    raw_entry = raw_manifest["entries"][0]

    assert summary["sample_count"] == 1
    assert summary["external_harness"]["harbor"]["sample_count"] == 1
    assert samples[0]["sample"]["eval_result"]["harbor_resolve_rate"] == 1.0
    assert raw_manifest["schema_version"] == "gage.external_harness.raw_archive.v1"
    assert raw_entry["task_id"] == "tb2_one_case"
    assert raw_entry["adapter_id"] == "harbor_tb2"
    assert raw_entry["provider"] == "harbor"
    assert raw_entry["artifacts"]["workdir_ref"] == "tb2_one_case/harbor_tb2"
    assert raw_entry["artifacts"]["launcher_result_ref"] == "tb2_one_case/harbor_tb2/launcher_result.json"
    assert raw_entry["artifacts"]["jobs_dir_ref"] == "tb2_one_case/harbor_tb2/jobs"
    assert raw_entry["artifacts"]["job_dir_ref"].startswith("tb2_one_case/harbor_tb2/jobs/")
    assert (provider_root / "invocation.json").exists()
    assert (provider_root / "job_config.json").exists()
    assert (provider_root / "harbor_job.json").exists()
    assert raw_job_tree.exists()
    assert (artifact_root / "infra" / "sample_record.json").exists()
    assert (artifact_root / "infra" / "harbor_job_result.json").exists()
    assert (artifact_root / "trials" / "trial_0001" / "infra" / "harbor_raw_result.json").exists()
    assert (artifact_root / "trials" / "trial_0001" / "agent" / "trajectory.json").exists()


def test_fixture_yaml_uses_lmstudio_litellm_endpoint_and_model_name() -> None:
    text = LMSTUDIO_FIXTURE.read_text(encoding="utf-8")
    payload = _load_payload(LMSTUDIO_FIXTURE)
    config = PipelineConfig.from_dict(payload)
    backend = config.backends[0]

    assert backend.type == "litellm"
    assert backend.config["api_base"] == "http://127.0.0.1:1234/v1"
    assert backend.config["model"] == "lm_studio/qwen/qwen3.5-9b"
    assert "openai_compatible" not in text
    assert "model: qwen/qwen3.5-9b" not in text
    assert "step_id:" not in text
    assert "role_adapter_id:" not in text


def test_lmstudio_fixture_keeps_pre_pulled_docker_image(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict(_load_payload(LMSTUDIO_FIXTURE))
    adapter = _base_agent_adapter(config)

    plan = adapter.translate(
        TaskBatchHarnessRequest(
            adapter_id="harbor_tb2",
            payload=_request_payload(config, tmp_path=tmp_path),
        )
    )

    assert plan.payload["job_config"]["environment"]["delete"] is False


def test_local_task_path_is_resolved_for_harbor_launcher_workdir(tmp_path: Path) -> None:
    config = PipelineConfig.from_dict(_load_payload(LMSTUDIO_FIXTURE))
    adapter = _base_agent_adapter(config)

    plan = adapter.translate(
        TaskBatchHarnessRequest(
            adapter_id="harbor_tb2",
            payload=_request_payload(config, tmp_path=tmp_path),
        )
    )

    task_path = Path(plan.payload["job_config"]["tasks"][0]["path"])
    assert task_path.is_absolute()
    assert task_path == (Path.cwd() / "tests/data/external_harness_kits/terminal_bench/gpt2-codegolf")


def test_relative_run_paths_are_resolved_before_launcher_subprocess() -> None:
    config = PipelineConfig.from_dict(_load_payload(LMSTUDIO_FIXTURE))
    adapter = _base_agent_adapter(config)
    payload = _request_payload(config, tmp_path=Path("runs/relative-harbor-smoke"))

    plan = adapter.translate(
        TaskBatchHarnessRequest(
            adapter_id="harbor_tb2",
            payload=payload,
        )
    )

    invocation = plan.payload["invocation"]
    assert invocation.workdir.is_absolute()
    assert invocation.jobs_dir.is_absolute()
    assert invocation.job_config_path.is_absolute()
    assert Path(plan.payload["job_config"]["jobs_dir"]).is_absolute()
    assert Path(invocation.launcher_argv[-1]).is_absolute()


def test_installed_client_mock_fixture_translates_and_launches_with_trial_side_endpoint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = _load_payload(INSTALLED_CLIENT_FIXTURE)
    config = PipelineConfig.from_dict(payload)
    adapter = _installed_client_adapter(config)
    monkeypatch.setattr(
        harbor_module.harbor_launcher,
        "run_launcher_subprocess",
        fake_harbor_launcher.run_fake_launcher,
    )

    plan = adapter.translate(
        TaskBatchHarnessRequest(
            adapter_id="harbor_tb2",
            payload=_request_payload(config, tmp_path=tmp_path),
        )
    )
    agent = plan.payload["job_config"]["agents"][0]
    handle = adapter.launch(plan)
    result = adapter.poll_until_done(handle)

    assert agent["env"]["OPENAI_BASE_URL"] == "http://host.docker.internal:1234/v1"
    assert "api_base" not in agent.get("kwargs", {})
    assert Path(result.payload["job_dir"]).exists()
    assert Path(result.payload["launcher_result_path"]).exists()


def test_swebench_pro_config_translates_with_builtin_installed_swe_agent(tmp_path: Path) -> None:
    payload = load_pipeline_config_payload(SWEBENCH_PRO_CONFIG)
    config = PipelineConfig.from_dict(payload)
    spec = config.role_adapters[0]
    adapter = HarborAdapter(
        adapter_id=spec.adapter_id,
        backend_id=spec.backend_id,
        env_id=spec.env_id,
        trial_policy=spec.trial_policy,
        params=spec.params,
    )

    plan = adapter.translate(
        TaskBatchHarnessRequest(
            adapter_id=spec.adapter_id,
            payload=_request_payload(config, tmp_path=tmp_path),
        )
    )

    agent = plan.payload["job_config"]["agents"][0]
    dataset = plan.payload["job_config"]["datasets"][0]
    invocation = plan.payload["invocation"]
    assert plan.payload["job_config"]["tasks"] == []
    assert dataset["name"] == "swebenchpro"
    assert dataset["version"] == "1.0"
    assert dataset["task_names"] == [SWEBENCH_PRO_ANSIBLE_TASK]
    assert dataset["n_tasks"] == 1
    expected_registry_path = (
        Path.cwd() / "tests/data/external_harness_kits/registries/harbor_swebenchpro_ansible_1case.json"
    )
    assert Path(dataset["registry_path"]) == expected_registry_path
    assert Path(dataset["registry_path"]).is_absolute()
    assert agent["name"] == "swe-agent"
    assert agent["model_name"] == "qwen/qwen3.5-9b"
    assert agent["kwargs"]["per_instance_call_limit"] == 200
    assert agent["env"]["OPENAI_BASE_URL"] == "http://host.docker.internal:1234/v1"
    assert agent["env"]["CONDA_DEFAULT_ENV"] == ""
    assert invocation.environ["OPENAI_BASE_URL"] == "http://host.docker.internal:1234/v1"
    assert invocation.environ["CONDA_DEFAULT_ENV"] == ""


def test_raises_installed_client_incompatible_without_trial_side_endpoint(
    tmp_path: Path,
) -> None:
    payload = _load_payload(INSTALLED_CLIENT_FIXTURE)
    del payload["role_adapters"][0]["params"]["harness"]["agent"]["extra_env"]
    config = PipelineConfig.from_dict(payload)
    adapter = _installed_client_adapter(config)

    with pytest.raises(ExternalHarnessError) as exc_info:
        adapter.translate(
            TaskBatchHarnessRequest(
                adapter_id="harbor_tb2",
                payload=_request_payload(config, tmp_path=tmp_path),
            )
        )

    assert exc_info.value.code == "external_harness.translate.installed_client_incompatible"


def _load_payload(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _resource_profile() -> ResourceProfile:
    return ResourceProfile([NodeResource(node_id="local", gpus=0, cpus=1)])


def _trace(run_id: str):
    from gage_eval.observability.trace import ObservabilityTrace

    return ObservabilityTrace(run_id=run_id)


def _installed_client_adapter(config: PipelineConfig) -> HarborAdapter:
    spec = config.role_adapters[0]
    return HarborAdapter(
        adapter_id=spec.adapter_id,
        backend_id=spec.backend_id,
        env_id=spec.env_id,
        trial_policy=spec.trial_policy,
        params=spec.params,
        installed_client_probe=lambda **_kwargs: True,
    )


def _base_agent_adapter(config: PipelineConfig) -> HarborAdapter:
    spec = config.role_adapters[0]
    return HarborAdapter(
        adapter_id=spec.adapter_id,
        backend_id=spec.backend_id,
        env_id=spec.env_id,
        trial_policy=spec.trial_policy,
        params=spec.params,
        model_endpoint_probe=lambda **_kwargs: True,
        docker_probe=lambda: True,
    )


def _request_payload(config: PipelineConfig, *, tmp_path: Path) -> dict:
    task = config.tasks[0]
    adapter_id = config.role_adapters[0].adapter_id
    archive_root = tmp_path / "external_harness"
    workdir = archive_root / task.task_id / adapter_id
    return {
        "run_id": "task15-installed-client-mock",
        "external_harness_root": str(archive_root),
        "external_harness_manifest_path": str(archive_root / "manifest.json"),
        "workdir": str(workdir),
        "jobs_dir": str(workdir / "jobs"),
        "job_config_path": str(workdir / "harbor_job.json"),
        "task": deepcopy(task.__dict__),
        "dataset": deepcopy(config.datasets[0].__dict__),
        "role_adapter": deepcopy(config.role_adapters[0].__dict__),
        "datasets": [deepcopy(dataset.__dict__) for dataset in config.datasets],
        "backends": [deepcopy(backend.__dict__) for backend in config.backends],
        "environments": deepcopy([item.to_dict() for item in config.environments]),
    }
