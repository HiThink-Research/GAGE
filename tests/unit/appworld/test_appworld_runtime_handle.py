from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop import _build_loop_inputs
from gage_eval.agent_eval_kits.appworld.sub_workflows.installed_client import _prepare_inputs
from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime


class _StubProvider:
    def __init__(self, runtime_handle: dict[str, Any]) -> None:
        self._handle = SimpleNamespace(runtime_handle=runtime_handle, sandbox=None)

    def get_handle(self):
        return self._handle


@pytest.mark.fast
def test_appworld_runtime_uses_runtime_handle_endpoints() -> None:
    calls: list[tuple[str, str, dict[str, Any], int]] = []

    def requester(
        endpoint: str,
        method: str,
        payload: dict[str, Any],
        timeout_s: int,
    ) -> dict[str, Any]:
        calls.append((endpoint, method, payload, timeout_s))
        if method == "initialize":
            return {
                "output": {
                    "method": method,
                    "instruction": "What is the title of the most-liked song in my Spotify playlists.",
                }
            }
        return {"output": {"method": method}}

    runtime = AppWorldRuntime(requester=requester)
    provider = _StubProvider(
        {
            "env_endpoint": "http://runtime-env",
            "apis_endpoint": "http://runtime-apis",
        }
    )
    sample = {"metadata": {"appworld": {"task_id": "task-1"}}}

    bootstrap = runtime.bootstrap(
        session=SimpleNamespace(),
        sample=sample,
        payload={},
        sandbox_provider=provider,
    )
    runtime.save(sample=sample, sandbox_provider=provider)

    assert calls[0][0] == "http://runtime-env"
    assert calls[0][2]["remote_apis_url"] == "http://runtime-apis"
    assert calls[1][0] == "http://runtime-env"
    assert calls[1][2]["task_id"] == "task-1"
    assert bootstrap["prompt_context"]["instruction"] == (
        "What is the title of the most-liked song in my Spotify playlists."
    )


@pytest.mark.fast
def test_appworld_workflows_project_runtime_instruction() -> None:
    session = SimpleNamespace(
        prompt_context={"instruction": "Solve the Spotify task."},
        runtime_context={"initialize": {"instruction": "Solve the Spotify task."}},
    )
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Complete the AppWorld task using available APIs."}],
            }
        ],
        "metadata": {"appworld": {"task_id": "task-1"}},
    }

    loop_inputs = _build_loop_inputs(session=session, sample=sample, payload={})
    client_inputs = _prepare_inputs(
        session=session,
        sample=sample,
        payload={},
        sandbox_provider=None,
    )

    assert "Use the provided AppWorld tools" in loop_inputs["messages"][0]["content"][0]["text"]
    assert loop_inputs["messages"][1]["content"][0]["text"] == "Solve the Spotify task."
    assert client_inputs["instruction"].startswith("Solve the Spotify task.")
    assert "Tool-use contract" in client_inputs["instruction"]
