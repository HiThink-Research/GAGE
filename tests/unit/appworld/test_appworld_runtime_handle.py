from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pytest

from gage_eval.role.agent.hooks import AgentHookContext
from gage_eval.sandbox.integrations.appworld.hooks import AppWorldInitializeHook, AppWorldSaveHook


@pytest.mark.fast
def test_appworld_hooks_use_runtime_handle_endpoints() -> None:
    calls: List[Tuple[str, Dict[str, Any]]] = []

    def requester(method: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        calls.append((method, payload))
        return {"output": {"method": method}}

    context = AgentHookContext(
        sample={},
        metadata={"appworld": {"task_id": "task-1"}},
        runtime_handle={
            "env_endpoint": "http://runtime-env",
            "apis_endpoint": "http://runtime-apis",
        },
        sandbox_config={
            "runtime_configs": {
                "env_endpoint": "http://config-env",
                "apis_endpoint": "http://config-apis",
            }
        },
    )

    init_hook = AppWorldInitializeHook(requester=requester)
    save_hook = AppWorldSaveHook(requester=requester)
    init_hook.run(context)
    save_hook.run(context)

    assert calls[0][0] == "initialize"
    assert calls[0][1]["remote_apis_url"] == "http://runtime-apis"
    assert calls[1][0] == "save"
    assert calls[1][1]["task_id"] == "task-1"
