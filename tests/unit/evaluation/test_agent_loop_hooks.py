from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gage_eval.role.agent.hooks import AgentHookContext
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope


class FakeBackend:
    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": "done"}


class FakeSandbox:
    def __init__(self, runtime_configs: Dict[str, Any] | None = None, resources: Dict[str, Any] | None = None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}

    def start(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {"env_endpoint": "http://env"}

    def teardown(self) -> None:
        return None

    def is_alive(self, timeout_s: float | None = None) -> bool:
        return True


class RecordingHook:
    def __init__(self, label: str, calls: List[tuple[str, AgentHookContext]]) -> None:
        self._label = label
        self._calls = calls

    def run(self, context: AgentHookContext) -> Dict[str, Any]:
        self._calls.append((self._label, context))
        context.hook_state.setdefault("order", []).append(self._label)
        return {"label": self._label}


@pytest.mark.fast
def test_agent_loop_runs_pre_and_post_hooks() -> None:
    manager = SandboxManager()
    manager.register_runtime("fake", FakeSandbox)
    provider = SandboxProvider(
        manager,
        {"runtime": "fake"},
        SandboxScope(run_id="run", task_id="task", sample_id="sample"),
    )
    calls: List[tuple[str, AgentHookContext]] = []
    loop = AgentLoop(
        backend=FakeBackend(),
        tool_router=ToolRouter(),
        pre_hooks=[RecordingHook("pre", calls)],
        post_hooks=[RecordingHook("post", calls)],
    )
    result = loop.run(
        messages=[{"role": "user", "content": "hi"}],
        sandbox_config={"runtime": "fake"},
        sandbox_provider=provider,
        sample={"metadata": {"appworld": {"task_id": "task-1"}}},
    )
    provider.release()
    assert result["answer"] == "done"
    assert [entry[0] for entry in calls] == ["pre", "post"]
    assert calls[0][1].runtime_handle["env_endpoint"] == "http://env"
    assert calls[1][1].agent_trace
    assert calls[1][1].hook_state["order"] == ["pre", "post"]
