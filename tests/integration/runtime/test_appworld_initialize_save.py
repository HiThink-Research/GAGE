from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.sandbox.integrations.appworld.hooks import AppWorldInitializeHook, AppWorldSaveHook
from gage_eval.sandbox.manager import SandboxHandle


class FakeBackend:
    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return {"answer": "done"}


class StubSandbox:
    def is_alive(self, timeout_s: float | None = None) -> bool:
        return True


class StubProvider:
    def __init__(self, runtime_handle: Dict[str, Any]) -> None:
        self._handle = SandboxHandle(
            sandbox=StubSandbox(),
            config={},
            runtime_handle=runtime_handle,
        )

    def get_handle(self) -> SandboxHandle:
        return self._handle


@pytest.mark.fast
def test_appworld_initialize_and_save_hooks() -> None:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def requester(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((method, payload))
        return {"output": {"method": method, "task_id": payload.get("task_id")}}

    loop = AgentLoop(
        backend=FakeBackend(),
        tool_router=ToolRouter(),
        pre_hooks=[AppWorldInitializeHook(requester=requester)],
        post_hooks=[AppWorldSaveHook(requester=requester)],
    )
    loop.run(
        messages=[{"role": "user", "content": "hi"}],
        sandbox_config={
            "runtime_configs": {
                "env_endpoint": "http://env",
                "apis_endpoint": "http://127.0.0.1:9000",
            }
        },
        sandbox_provider=StubProvider(
            {
                "env_endpoint": "http://env",
                "apis_endpoint": "http://127.0.0.1:9000",
            }
        ),
        sample={"metadata": {"appworld": {"task_id": "task-1"}}},
    )

    assert calls[0][0] == "initialize"
    assert calls[0][1]["task_id"] == "task-1"
    assert calls[0][1]["remote_apis_url"] == "http://127.0.0.1:9000"
    assert calls[1][0] == "save"
