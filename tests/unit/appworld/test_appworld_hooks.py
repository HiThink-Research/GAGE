from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gage_eval.role.agent.hooks import AgentHookContext
from gage_eval.sandbox.integrations.appworld.hooks import AppWorldInitializeHook, AppWorldSaveHook


@pytest.mark.fast
def test_appworld_initialize_hook_builds_payload() -> None:
    calls: List[tuple[str, Dict[str, Any]]] = []

    def requester(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((method, payload))
        return {"output": {"task_id": payload.get("task_id"), "status": "ok"}}

    hook = AppWorldInitializeHook(requester=requester)
    context = AgentHookContext(
        sample={},
        metadata={"appworld": {"task_id": "task-1", "ground_truth_mode": "auto", "allowed_apps": ["mail"]}},
        runtime_handle={"env_endpoint": "http://env"},
        sandbox_config={"runtime_configs": {"apis_endpoint": "http://apis"}},
    )

    result = hook.run(context)
    assert result["task_id"] == "task-1"
    assert calls[0][0] == "initialize"
    assert calls[0][1]["task_id"] == "task-1"
    assert calls[0][1]["remote_apis_url"] == "http://apis"
    assert calls[0][1]["ground_truth_mode"] == "minimal"
    assert "allowed_apps" not in calls[0][1]
    assert context.hook_state["appworld"]["initialize"]["status"] == "ok"


@pytest.mark.fast
def test_appworld_save_hook_uses_task_id() -> None:
    calls: List[tuple[str, Dict[str, Any]]] = []

    def requester(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((method, payload))
        return {"output": None}

    hook = AppWorldSaveHook(requester=requester)
    context = AgentHookContext(
        sample={},
        metadata={"appworld": {"task_id": "task-2"}},
        runtime_handle={},
        sandbox_config={"runtime_configs": {"env_endpoint": "http://env"}},
        agent_trace=[],
    )

    hook.run(context)
    assert calls[0][0] == "save"
    assert calls[0][1]["task_id"] == "task-2"
