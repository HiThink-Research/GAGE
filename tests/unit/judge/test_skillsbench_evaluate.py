from __future__ import annotations

from pathlib import Path
import subprocess

from gage_eval.role.judge.skillsbench_evaluate import SkillsBenchEvaluate


def test_skillsbench_evaluate_uses_reward_file(monkeypatch, tmp_path: Path) -> None:
    workspace_source = tmp_path / "agent-workspace"
    workspace_source.mkdir(parents=True, exist_ok=True)
    (workspace_source / "app.py").write_text("print('ok')\n", encoding="utf-8")
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")
    verifier_workspace_dir = tmp_path / "verifier" / "workspace"
    verifier_logs_dir = tmp_path / "verifier" / "logs"
    stdout_path = tmp_path / "verifier" / "stdout.log"
    stderr_path = tmp_path / "verifier" / "stderr.log"

    monkeypatch.setattr(
        "gage_eval.role.judge.skillsbench_evaluate.ensure_docker_image",
        lambda runtime_configs: runtime_configs.get("image"),
    )
    monkeypatch.setattr(
        "gage_eval.role.judge.skillsbench_evaluate.build_docker_run_command",
        lambda **kwargs: ["docker", "run", "skillsbench:latest"],
    )

    seen_exec_commands: list[str] = []

    def fake_run(args, **kwargs):
        if args[:2] == ["docker", "run"]:
            return subprocess.CompletedProcess(args, 0, stdout="container-1\n", stderr="")
        if args[:2] == ["docker", "exec"]:
            if len(args) >= 6:
                seen_exec_commands.append(str(args[-1]))
        if args[:2] == ["docker", "cp"] and "container-1:/logs/verifier/." in args[2]:
            destination = Path(args[3])
            destination.mkdir(parents=True, exist_ok=True)
            (destination / "reward.txt").write_text("1\n", encoding="utf-8")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(args, 0, stdout="ok", stderr="")

    monkeypatch.setattr("gage_eval.role.judge.skillsbench_evaluate.subprocess.run", fake_run)

    result = SkillsBenchEvaluate().invoke(
        {
            "sample": {
                "id": "skillsbench__1",
                "sandbox": {"runtime_configs": {"image": "skillsbench:latest"}},
                "resources": {"cpu": 1},
                "metadata": {
                    "skillsbench": {
                        "task_id": "skillsbench__1",
                        "image": "skillsbench:latest",
                        "workdir": "/app",
                        "tests_dir": str(tests_dir),
                    }
                },
            },
            "artifact_paths": {
                "agent_workspace_dir": str(workspace_source),
                "verifier_workspace_dir": str(verifier_workspace_dir),
                "verifier_logs_dir": str(verifier_logs_dir),
                "verifier_stdout_file": str(stdout_path),
                "verifier_stderr_file": str(stderr_path),
            },
        }
    )

    assert result["resolved"] is True
    assert result["score"] == 1.0
    assert stdout_path.exists()
    assert stderr_path.exists()
    assert (verifier_logs_dir / "reward.txt").exists()
    assert any("ln -sfn /app/output /output" in command for command in seen_exec_commands)


def test_skillsbench_evaluate_fails_without_workspace(tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\nexit 0\n", encoding="utf-8")

    result = SkillsBenchEvaluate().invoke(
        {
            "sample": {
                "id": "skillsbench__missing",
                "metadata": {
                    "skillsbench": {
                        "task_id": "skillsbench__missing",
                        "image": "skillsbench:latest",
                        "workdir": "/app",
                        "tests_dir": str(tests_dir),
                    }
                },
            },
            "artifact_paths": {},
        }
    )

    assert result["resolved"] is False
    assert result["failure_reason"] == "missing_agent_workspace"
