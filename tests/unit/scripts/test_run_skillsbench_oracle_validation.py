from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "run_skillsbench_oracle_validation.py"
    spec = importlib.util.spec_from_file_location("skillsbench_oracle_validation_script", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_output_dir_defaults_under_workdir() -> None:
    module = _load_module()

    assert module._resolve_output_dir("/app") == "/app/output"
    assert module._resolve_output_dir("/workspace/") == "/workspace/output"
    assert module._resolve_output_dir("/") == "/output"


def test_prepare_solution_mounts_command_mounts_solution_and_output() -> None:
    module = _load_module()

    command = module._prepare_solution_mounts_command("/app")

    assert "rm -rf /solution" in command
    assert "mkdir -p /solution" in command
    assert "mkdir -p /app/output" in command
    assert "ln -sfn /app/output /output" in command


def test_select_shard_splits_samples_evenly() -> None:
    module = _load_module()
    samples = [{"id": f"task-{idx}"} for idx in range(7)]

    assert module._select_shard(samples, shard_index=0, shard_count=3) == [
        {"id": "task-0"},
        {"id": "task-3"},
        {"id": "task-6"},
    ]
    assert module._select_shard(samples, shard_index=1, shard_count=3) == [
        {"id": "task-1"},
        {"id": "task-4"},
    ]
    assert module._select_shard(samples, shard_index=2, shard_count=3) == [
        {"id": "task-2"},
        {"id": "task-5"},
    ]


def test_select_shard_validates_bounds() -> None:
    module = _load_module()

    try:
        module._select_shard([], shard_index=0, shard_count=0)
    except ValueError as exc:
        assert "shard_count" in str(exc)
    else:
        raise AssertionError("expected shard_count validation error")

    try:
        module._select_shard([], shard_index=2, shard_count=2)
    except ValueError as exc:
        assert "shard_index" in str(exc)
    else:
        raise AssertionError("expected shard_index validation error")
