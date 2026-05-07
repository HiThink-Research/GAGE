from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime


@pytest.mark.fast
def test_appworld_runtime_builds_initialize_payload() -> None:
    calls: list[tuple[str, str, dict[str, Any], int]] = []

    def requester(
        endpoint: str,
        method: str,
        payload: dict[str, Any],
        timeout_s: int,
    ) -> dict[str, Any]:
        calls.append((endpoint, method, payload, timeout_s))
        return {"output": {"task_id": payload.get("task_id"), "status": "ok"}}

    runtime = AppWorldRuntime(requester=requester)
    sample = {
        "metadata": {
            "appworld": {
                "task_id": "task-1",
                "ground_truth_mode": "auto",
                "allowed_apps": ["mail"],
            }
        }
    }
    runtime_handle = {"env_endpoint": "http://env", "apis_endpoint": "http://apis"}

    result = runtime.bootstrap(
        session=SimpleNamespace(),
        sample=sample,
        payload={"runtime_handle": runtime_handle},
    )

    assert result["benchmark_state"]["initialize"]["task_id"] == "task-1"
    assert calls[0][1] == "initialize"
    assert calls[0][2]["task_id"] == "task-1"
    assert calls[0][2]["remote_apis_url"] == "http://apis"
    assert calls[0][2]["ground_truth_mode"] == "minimal"
    assert "allowed_apps" not in calls[0][2]


@pytest.mark.fast
def test_appworld_runtime_save_requires_task_id() -> None:
    runtime = AppWorldRuntime(requester=lambda *_: {"output": {}})

    with pytest.raises(ValueError, match="appworld.task_id is required for save"):
        runtime.save(
            sample={"metadata": {"appworld": {}}},
            payload={"runtime_handle": {"env_endpoint": "http://env"}},
        )
