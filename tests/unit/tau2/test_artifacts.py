from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_eval_kits.tau2.artifacts import persist_tau2_artifacts


class _FakeTau2Runtime:
    def get_state(self) -> dict[str, object]:
        return {
            "messages": [{"role": "assistant", "content": "done"}],
            "termination_reason": "agent_stop",
            "agent_cost": 0.1,
            "user_cost": 0.0,
        }


class _FakeProvider:
    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=_FakeTau2Runtime())


class _FakePydanticMessage:
    def __init__(self, role: str, content: str) -> None:
        self.role = role
        self.content = content

    def model_dump(self, mode: str = "python") -> dict[str, str]:
        assert mode in {"python", "json"}
        return {"role": self.role, "content": self.content}

    def __str__(self) -> str:
        return f"SHOULD_NOT_USE_STR({self.role}:{self.content})"


def _build_session(tmp_path: Path) -> SimpleNamespace:
    sample_root = tmp_path / "sample"
    artifacts_dir = sample_root / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        artifact_layout={
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
        }
    )


@pytest.mark.fast
def test_persist_tau2_artifacts_exports_state_trajectory_and_cost(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    artifact_paths = persist_tau2_artifacts(
        session=session,
        scheduler_output={"agent_trace": [{"tool_name": "respond"}]},
        sandbox_provider=_FakeProvider(),
    )

    assert artifact_paths == {
        "tau2_state": "artifacts/tau2_state.json",
        "tau2_trajectory": "artifacts/tau2_trajectory.json",
        "tau2_cost": "artifacts/tau2_cost.json",
    }
    trajectory_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_trajectory.json").read_text())
    cost_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_cost.json").read_text())

    assert trajectory_payload["source"] == "runtime_state.messages"
    assert trajectory_payload["events"] == [{"role": "assistant", "content": "done"}]
    assert cost_payload == {
        "agent_cost": 0.1,
        "user_cost": 0.0,
        "agent_total_tokens": None,
        "user_total_tokens": None,
        "termination_reason": "agent_stop",
    }


@pytest.mark.fast
def test_persist_tau2_artifacts_falls_back_to_agent_trace_without_runtime_messages(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_tau2_artifacts(
        session=session,
        scheduler_output={"agent_trace": [{"tool_name": "respond"}]},
        sandbox_provider=None,
    )

    trajectory_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_trajectory.json").read_text())

    assert trajectory_payload["source"] == "agent_trace"
    assert trajectory_payload["events"] == [{"tool_name": "respond"}]


@pytest.mark.fast
def test_persist_tau2_artifacts_redacts_secret_text_in_fallback_artifacts(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_tau2_artifacts(
        session=session,
        scheduler_output={
            "agent_trace": [
                {
                    "tool_name": "respond",
                    "content": "Authorization: Bearer abc123",
                }
            ]
        },
        sandbox_provider=None,
    )

    serialized = (tmp_path / "sample" / "artifacts" / "tau2_trajectory.json").read_text(encoding="utf-8")
    assert "Bearer abc123" not in serialized
    assert "<redacted:" in serialized


@pytest.mark.fast
def test_persist_tau2_artifacts_fills_missing_terminal_diagnostics(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    persist_tau2_artifacts(
        session=session,
        scheduler_output={"agent_trace": []},
        sandbox_provider=None,
    )

    state_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text())
    cost_payload = json.loads((tmp_path / "sample" / "artifacts" / "tau2_cost.json").read_text())

    assert state_payload["termination_reason"] == "too_many_errors"
    assert state_payload["agent_cost"] == 0.0
    assert state_payload["user_cost"] == 0.0
    assert cost_payload["termination_reason"] == "too_many_errors"


@pytest.mark.fast
def test_persist_tau2_artifacts_uses_trial_agent_owner_and_model_dump(tmp_path: Path) -> None:
    sink = RuntimeArtifactSink(base_dir=str(tmp_path))
    session = SimpleNamespace(
        run_id="run-1",
        task_id="task-1",
        sample_id="sample-1",
        artifact_sink=sink,
        runtime_context={"trial_id": "trial_0001"},
        scheduler_state={},
        artifact_layout={"sample_root": str(tmp_path / "legacy-sample")},
    )

    class Runtime:
        def get_state(self) -> dict[str, object]:
            return {
                "messages": [_FakePydanticMessage("assistant", "hello")],
                "termination_reason": "agent_stop",
            }

    paths = persist_tau2_artifacts(
        session=session,
        scheduler_output={},
        environment_lease=SimpleNamespace(environment=Runtime()),
    )

    assert paths == {
        "tau2_state": "artifacts/task-1/sample-1/trials/trial_0001/agent/tau2_state.json",
        "tau2_trajectory": "artifacts/task-1/sample-1/trials/trial_0001/agent/tau2_trajectory.json",
        "tau2_cost": "artifacts/task-1/sample-1/trials/trial_0001/agent/tau2_cost.json",
    }
    state_path = tmp_path / "run-1" / paths["tau2_state"]
    trajectory_path = tmp_path / "run-1" / paths["tau2_trajectory"]
    state_text = state_path.read_text(encoding="utf-8")
    trajectory = json.loads(trajectory_path.read_text(encoding="utf-8"))
    assert "SHOULD_NOT_USE_STR" not in state_text
    assert trajectory["events"] == [{"role": "assistant", "content": "hello"}]
    assert not (tmp_path / "run-1" / "artifacts/task-1/sample-1/tau2_state.json").exists()
