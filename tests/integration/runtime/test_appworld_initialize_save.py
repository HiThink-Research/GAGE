from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime


class _StubProvider:
    def __init__(self, runtime_handle: dict[str, Any]) -> None:
        self._handle = SimpleNamespace(runtime_handle=runtime_handle, sandbox=None)

    def get_handle(self):
        return self._handle


@pytest.mark.fast
def test_appworld_runtime_bootstrap_and_save_without_legacy_hooks() -> None:
    calls: list[tuple[str, str, dict[str, Any], int]] = []

    def requester(
        endpoint: str,
        method: str,
        payload: dict[str, Any],
        timeout_s: int,
    ) -> dict[str, Any]:
        calls.append((endpoint, method, payload, timeout_s))
        return {"output": {"method": method, "task_id": payload.get("task_id")}}

    runtime = AppWorldRuntime(requester=requester)
    provider = _StubProvider(
        {
            "env_endpoint": "http://env",
            "apis_endpoint": "http://127.0.0.1:9000",
            "mcp_endpoint": "http://127.0.0.1:5001",
        }
    )
    sample = {
        "metadata": {
            "appworld": {
                "task_id": "task-1",
                "ground_truth_mode": "auto",
                "allowed_apps": ["mail"],
            }
        }
    }

    bootstrap = runtime.bootstrap(
        session=SimpleNamespace(),
        sample=sample,
        payload={},
        sandbox_provider=provider,
    )
    saved = runtime.save(sample=sample, sandbox_provider=provider)

    assert calls[0][0] == "http://env"
    assert calls[0][1] == "initialize"
    assert calls[0][2]["task_id"] == "task-1"
    assert calls[0][2]["remote_apis_url"] == "http://127.0.0.1:9000"
    assert calls[0][2]["remote_mcp_url"] == "http://127.0.0.1:5001"
    assert calls[0][2]["ground_truth_mode"] == "minimal"
    assert calls[1][1] == "save"
    assert bootstrap["benchmark_state"]["initialize"]["method"] == "initialize"
    assert saved["method"] == "save"
