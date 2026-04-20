from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

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
        "termination_reason": "agent_stop",
    }


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
