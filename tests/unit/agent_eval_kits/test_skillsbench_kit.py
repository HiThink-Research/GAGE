from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from gage_eval.agent_eval_kits.skillsbench.judge_bridge import build_verifier_input
from gage_eval.agent_eval_kits.skillsbench.kit import build_kit
from gage_eval.agent_eval_kits.skillsbench.sub_workflow import (
    capture_environment_artifacts,
    finalize_result,
    prepare_environment,
    prepare_inputs,
)
from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
from gage_eval.config.pipeline_config import PipelineConfig


def _sample() -> dict:
    return {
        "id": "skillsbench__1",
        "instruction": "Fix the task",
        "cwd": "/app",
        "metadata": {
            "skillsbench": {
                "task_id": "skillsbench__1",
                "category": "devops",
                "difficulty": "medium",
            }
        },
    }


def test_skillsbench_build_kit_returns_definition() -> None:
    kit = build_kit()

    assert kit.kit_id == "skillsbench"
    assert kit.verifier_kind == "judge_adapter"
    assert kit.required_surfaces == ("terminal", "fs")


def test_skillsbench_prepare_inputs() -> None:
    session = SimpleNamespace(
        trace=SimpleNamespace(run_id="run-1"),
        plan=SimpleNamespace(benchmark_kit_id="skillsbench"),
        artifacts=ArtifactLayout.for_sample("/tmp/runs", "run-1", "skillsbench__1"),
    )

    payload = prepare_inputs(_sample(), session)

    assert payload["sample_id"] == "skillsbench__1"
    assert payload["instruction"] == "Fix the task"
    assert payload["cwd"] == "/app"
    assert payload["artifacts"]["patch_file"].endswith("submission.patch")


def test_skillsbench_build_verifier_input_uses_scheduler_artifacts() -> None:
    artifacts = ArtifactLayout.for_sample("/tmp/runs", "run-1", "skillsbench__1")
    scheduler_result = SimpleNamespace(
        status="success",
        answer="done",
        patch_path=str(Path(artifacts.patch_file)),
        stdout_path=str(Path(artifacts.stdout_file)),
        trajectory_path=str(Path(artifacts.trajectory_file)),
        artifacts={"agent_workspace_dir": "/tmp/runs/run-1/skillsbench__1/workspace"},
        raw_output={},
    )

    verifier_input = build_verifier_input(_sample(), scheduler_result, artifacts)

    assert verifier_input.benchmark_kit_id == "skillsbench"
    assert verifier_input.artifact_paths["agent_workspace_dir"].endswith("workspace")


def test_skillsbench_finalize_result_includes_verifier_input() -> None:
    artifacts = ArtifactLayout.for_sample("/tmp/runs", "run-1", "skillsbench__1")
    scheduler_result = SimpleNamespace(status="success", answer="done", raw_output={})

    result = finalize_result(_sample(), scheduler_result, artifacts)

    assert result["sample_id"] == "skillsbench__1"
    assert result["artifact_paths"]["sample_file"].endswith("sample.json")
    assert "verifier_input" not in result


def test_skillsbench_prepare_environment_prefers_host_codex_home() -> None:
    calls: list[dict[str, object]] = []

    class _FakeEnvironment:
        def exec(self, command: str, *, cwd: str | None = None, env=None, timeout_sec: int = 30):
            calls.append(
                {
                    "command": command,
                    "cwd": cwd,
                    "env": env,
                    "timeout_sec": timeout_sec,
                }
            )
            return SimpleNamespace(exit_code=0)

    session = SimpleNamespace(
        artifacts=ArtifactLayout.for_sample("/tmp/runs", "run-1", "skillsbench__1"),
    )
    request = SimpleNamespace(metadata={"code_home": "/root/.codex"})
    sample = _sample()
    sample["metadata"]["skillsbench"]["agent_timeout_sec"] = 321

    result = prepare_environment(sample, session, _FakeEnvironment(), request=request)

    assert result["prepare_environment_exit_code"] == 0
    assert calls
    command = str(calls[0]["command"])
    assert 'GAGE_CODEX_HOST_HOME' in command
    assert 'cp -R "${GAGE_CODEX_HOST_HOME}/." "$CODEX_HOME/"' in command
    assert 'if [ ! -f "$CODEX_HOME/auth.json" ] && [ -n "${OPENAI_API_KEY:-}" ]; then' in command
    assert calls[0]["cwd"] == "/app"
    assert calls[0]["timeout_sec"] == 321


def test_skillsbench_capture_environment_artifacts_prefers_docker_cp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    class _FakeEnvironment:
        def runtime_handle(self):
            return {"container_id": "container-1"}

        def exec(self, command: str, *, cwd: str | None = None, env=None, timeout_sec: int = 30):
            raise AssertionError("docker cp fast path should skip tar fallback")

    def fake_run(args, **kwargs):
        calls.append([str(arg) for arg in args])
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("gage_eval.agent_eval_kits.skillsbench.sub_workflow.subprocess.run", fake_run)
    session = SimpleNamespace(
        artifacts=ArtifactLayout.for_sample("/tmp/runs", "run-1", "skillsbench__1"),
    )
    sample = _sample()
    sample["sandbox"] = {"runtime_configs": {"docker_bin": "docker"}}

    result = capture_environment_artifacts(sample, session, _FakeEnvironment(), client_result=None)

    assert result["agent_workspace_dir"].endswith("agent/workspace")
    assert calls == [
        [
            "docker",
            "cp",
            "container-1:/app/.",
            str(Path(result["agent_workspace_dir"])),
        ]
    ]


@pytest.mark.io
def test_skillsbench_smoke_config_is_parseable() -> None:
    config_path = (
        Path(__file__).resolve().parents[3]
        / "config"
        / "custom"
        / "skillsbench"
        / "smoke_installed_client_skillsbench.yaml"
    )
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = PipelineConfig.from_dict(payload)

    assert payload["metadata"]["name"] == "smoke_installed_client_skillsbench"
    assert payload["agent_runtimes"][0]["benchmark_kit_id"] == "skillsbench"
    assert payload["benchmark_kits"][0]["kit_id"] == "skillsbench"
    assert config.metadata["name"] == "smoke_installed_client_skillsbench"
