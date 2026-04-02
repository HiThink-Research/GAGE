from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

from gage_eval.assets.datasets.loaders.skillsbench_loader import SkillsBenchHarborLoader
from gage_eval.config.pipeline_config import DatasetSpec


def _write_task(root: Path, name: str, *, task_toml: str, dockerfile: str) -> Path:
    task_dir = root / "datasets" / "skillsbench" / name
    (task_dir / "environment").mkdir(parents=True, exist_ok=True)
    (task_dir / "tests").mkdir(parents=True, exist_ok=True)
    (task_dir / "solution").mkdir(parents=True, exist_ok=True)
    (task_dir / "instruction.md").write_text("Solve the task.", encoding="utf-8")
    (task_dir / "task.toml").write_text(task_toml, encoding="utf-8")
    (task_dir / "environment" / "Dockerfile").write_text(dockerfile, encoding="utf-8")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    (task_dir / "solution" / "README.md").write_text("solution", encoding="utf-8")
    return task_dir


@pytest.mark.io
def test_skillsbench_loader_builds_sample_with_runtime_and_verifier_env(tmp_path: Path) -> None:
    repo_root = tmp_path / "harbor-datasets"
    _write_task(
        repo_root,
        "alias-env-task",
        task_toml="""
version = "1.0"

[metadata]
difficulty = "medium"
category = "devops"

[agent]
timeout_sec = 123

[verifier]
timeout_sec = 456

[verifier.env]
GITHUB_TOKEN = "${GH_AUTH_TOKEN}"

[solution.env]
GH_AUTH_TOKEN = "${GH_AUTH_TOKEN}"

[environment]
build_timeout_sec = 600
cpus = 2
memory_mb = 4096

[environment.env]
OPENAI_API_KEY = "${OPENAI_API_KEY}"
""".strip(),
        dockerfile="""
FROM python:3.12-slim
ENV WORKSPACE=/root/project
WORKDIR ${WORKSPACE}
""".strip(),
    )
    spec = DatasetSpec(
        dataset_id="skillsbench_loader_smoke",
        loader="skillsbench_harbor",
        params={
            "local_repo_dir": str(repo_root),
            "generated_dir": str(tmp_path / "generated"),
            "auto_download": False,
        },
    )

    source = SkillsBenchHarborLoader(spec).load(None)
    records = list(source.records)

    assert len(records) == 1
    sample = asdict(records[0]) if is_dataclass(records[0]) else records[0]
    assert sample["metadata"]["workspace_root"] == "/root/project"
    runtime_configs = sample["sandbox"]["runtime_configs"]
    assert runtime_configs["cpus"] == 2
    assert runtime_configs["memory"] == "4096m"
    assert runtime_configs["env"]["OPENAI_API_KEY"] == "${OPENAI_API_KEY}"
    skillsbench_meta = sample["metadata"]["skillsbench"]
    assert skillsbench_meta["resource_limits"]["cpu"] == 2
    assert skillsbench_meta["runtime_env_sources"]["OPENAI_API_KEY"] == "OPENAI_API_KEY"
    assert skillsbench_meta["verifier_env_sources"]["GITHUB_TOKEN"] == "GH_AUTH_TOKEN"
    assert "GH_AUTH_TOKEN" in skillsbench_meta["missing_env_vars"]
    assert Path(skillsbench_meta["generated_dockerfile"]).exists()


@pytest.mark.io
def test_skillsbench_loader_can_skip_tasks_with_missing_env(tmp_path: Path) -> None:
    repo_root = tmp_path / "harbor-datasets"
    _write_task(
        repo_root,
        "requires-secret",
        task_toml="""
version = "1.0"

[verifier]
timeout_sec = 100

[solution.env]
GH_AUTH_TOKEN = "${GH_AUTH_TOKEN}"

[environment]
build_timeout_sec = 100
""".strip(),
        dockerfile="FROM python:3.12-slim\nWORKDIR /app\n",
    )
    spec = DatasetSpec(
        dataset_id="skillsbench_loader_skip_missing_env",
        loader="skillsbench_harbor",
        params={
            "local_repo_dir": str(repo_root),
            "generated_dir": str(tmp_path / "generated"),
            "auto_download": False,
            "skip_missing_env_tasks": True,
        },
    )

    source = SkillsBenchHarborLoader(spec).load(None)
    assert list(source.records) == []


@pytest.mark.io
def test_skillsbench_loader_can_raise_minimum_build_timeout(tmp_path: Path) -> None:
    repo_root = tmp_path / "harbor-datasets"
    _write_task(
        repo_root,
        "slow-build-task",
        task_toml="""
version = "1.0"

[environment]
build_timeout_sec = 600
""".strip(),
        dockerfile="FROM python:3.12-slim\nWORKDIR /app\n",
    )
    spec = DatasetSpec(
        dataset_id="skillsbench_loader_min_timeout",
        loader="skillsbench_harbor",
        params={
            "local_repo_dir": str(repo_root),
            "generated_dir": str(tmp_path / "generated"),
            "auto_download": False,
            "min_build_timeout_sec": 3600,
        },
    )

    source = SkillsBenchHarborLoader(spec).load(None)
    records = list(source.records)

    assert len(records) == 1
    sample = asdict(records[0]) if is_dataclass(records[0]) else records[0]
    runtime_configs = sample["sandbox"]["runtime_configs"]
    skillsbench_meta = sample["metadata"]["skillsbench"]
    assert runtime_configs["build_timeout_s"] == 3600
    assert skillsbench_meta["task_build_timeout_sec"] == 600
    assert skillsbench_meta["build_timeout_sec"] == 3600


@pytest.mark.io
def test_skillsbench_loader_mounts_host_codex_home_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo_root = tmp_path / "harbor-datasets"
    host_codex_home = tmp_path / ".codex"
    host_codex_home.mkdir(parents=True, exist_ok=True)
    (host_codex_home / "auth.json").write_text('{"provider":"chatgpt"}', encoding="utf-8")
    monkeypatch.setenv("GAGE_CODEX_HOST_HOME", str(host_codex_home))
    _write_task(
        repo_root,
        "codex-auth-task",
        task_toml="""
version = "1.0"

[environment]
build_timeout_sec = 600
""".strip(),
        dockerfile="FROM python:3.12-slim\nWORKDIR /app\n",
    )
    spec = DatasetSpec(
        dataset_id="skillsbench_loader_codex_mount",
        loader="skillsbench_harbor",
        params={
            "local_repo_dir": str(repo_root),
            "generated_dir": str(tmp_path / "generated"),
            "auto_download": False,
        },
    )

    source = SkillsBenchHarborLoader(spec).load(None)
    records = list(source.records)

    assert len(records) == 1
    sample = asdict(records[0]) if is_dataclass(records[0]) else records[0]
    runtime_configs = sample["sandbox"]["runtime_configs"]
    assert runtime_configs["env"]["GAGE_CODEX_HOST_HOME"] == "/gage-host-codex"
    assert runtime_configs["volumes"] == [f"{host_codex_home.resolve()}:/gage-host-codex:ro"]


@pytest.mark.io
def test_skillsbench_loader_sets_build_retry_defaults(tmp_path: Path) -> None:
    repo_root = tmp_path / "harbor-datasets"
    _write_task(
        repo_root,
        "retry-build-task",
        task_toml="""
version = "1.0"

[environment]
build_timeout_sec = 600
""".strip(),
        dockerfile="FROM python:3.12-slim\nWORKDIR /app\n",
    )
    spec = DatasetSpec(
        dataset_id="skillsbench_loader_build_retry_defaults",
        loader="skillsbench_harbor",
        params={
            "local_repo_dir": str(repo_root),
            "generated_dir": str(tmp_path / "generated"),
            "auto_download": False,
        },
    )

    source = SkillsBenchHarborLoader(spec).load(None)
    records = list(source.records)

    assert len(records) == 1
    sample = asdict(records[0]) if is_dataclass(records[0]) else records[0]
    runtime_configs = sample["sandbox"]["runtime_configs"]
    skillsbench_meta = sample["metadata"]["skillsbench"]
    assert runtime_configs["build_retry_attempts"] == 2
    assert runtime_configs["build_retry_backoff_s"] == 3.0
    assert skillsbench_meta["build_retry_attempts"] == 2
    assert skillsbench_meta["build_retry_backoff_s"] == 3.0
