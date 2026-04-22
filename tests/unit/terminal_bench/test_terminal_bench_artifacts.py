from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from gage_eval.agent_eval_kits.terminal_bench.artifacts import normalize_terminal_answer


class _FakeSandbox:
    def exec(self, command: str, timeout: int) -> SimpleNamespace:
        return SimpleNamespace(
            exit_code=0,
            stdout="hello.txt\t6\t2cf24dba5fb0a30e26e83b2ac5b9e29e\n",
            stderr="",
        )


class _FakeProvider:
    def get_handle(self) -> SimpleNamespace:
        return SimpleNamespace(sandbox=_FakeSandbox())


def _build_session(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        benchmark_state={
            "workspace_state": {
                "available": True,
                "cwd": "/workspace",
                "files": {},
            }
        },
        runtime_context={"cwd": "/workspace"},
    )


@pytest.mark.fast
def test_normalize_terminal_answer_returns_done_when_workspace_changes(tmp_path: Path) -> None:
    session = _build_session(tmp_path)

    normalized = normalize_terminal_answer(
        session=session,
        sample={"expected_answer": "done"},
        scheduler_output={"answer": "Created hello.txt successfully."},
        sandbox_provider=_FakeProvider(),
    )

    assert normalized == "done"
