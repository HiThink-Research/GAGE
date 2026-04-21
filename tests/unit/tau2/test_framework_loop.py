from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.tau2.sub_workflows.framework_loop import _finalize_loop_result


class _FakeRuntime:
    def __init__(
        self,
        *,
        initial_reason: str | None = None,
        initial_detail: str | None = None,
    ) -> None:
        self._termination_reason = initial_reason
        self._termination_detail = initial_detail
        self.mark_calls: list[str] = []

    def mark_agent_exhausted(self, detail: str) -> None:
        if self._termination_reason is None:
            self.mark_calls.append(detail)
            self._termination_reason = "agent_error"
            self._termination_detail = detail

    def get_state(self) -> dict[str, object]:
        return {
            "messages": [],
            "termination_reason": self._termination_reason,
            "termination_detail": self._termination_detail,
            "agent_cost": 0.0,
            "user_cost": 0.0,
        }


class _Provider:
    def __init__(self, runtime: _FakeRuntime) -> None:
        self.runtime = runtime

    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=self.runtime)


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
@pytest.mark.parametrize(
    "loop_exit_reason,detail",
    [
        ("tool_call_retry_budget", "no_tool_call_from_agent"),
        ("max_turns", "agent_loop_max_turns"),
    ],
)
def test_finalize_loop_result_marks_runtime_for_loop_exit_reason(
    tmp_path: Path,
    loop_exit_reason: str,
    detail: str,
) -> None:
    runtime = _FakeRuntime()
    session = _build_session(tmp_path)

    output = _finalize_loop_result(
        session=session,
        sample={},
        scheduler_output={"loop_exit_reason": loop_exit_reason},
        sandbox_provider=_Provider(runtime),
    )

    assert runtime.mark_calls == [detail]
    assert output["artifact_paths"]["tau2_state"] == "artifacts/tau2_state.json"
    state_payload = json.loads(
        (tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text()
    )
    assert state_payload["termination_detail"] == detail
    assert output["artifact_paths"]["tau2_cost"] == "artifacts/tau2_cost.json"


@pytest.mark.fast
def test_finalize_loop_result_does_not_override_existing_runtime_termination(tmp_path: Path) -> None:
    runtime = _FakeRuntime(initial_reason="user_stop", initial_detail="already_done")
    session = _build_session(tmp_path)

    _finalize_loop_result(
        session=session,
        sample={},
        scheduler_output={"loop_exit_reason": "tool_call_retry_budget"},
        sandbox_provider=_Provider(runtime),
    )

    assert not runtime.mark_calls
    state_payload = json.loads(
        (tmp_path / "sample" / "artifacts" / "tau2_state.json").read_text()
    )
    assert state_payload["termination_detail"] == "already_done"
