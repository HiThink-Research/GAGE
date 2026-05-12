from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop as framework_loop
from gage_eval.agent_eval_kits.appworld.sub_workflows.framework_loop import _build_loop_inputs
from gage_eval.agent_eval_kits.appworld.sub_workflows.installed_client import _prepare_inputs
from gage_eval.agent_eval_kits.appworld.runtime import AppWorldRuntime


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
    runtime_handle = {
        "env_endpoint": "http://runtime-env",
        "apis_endpoint": "http://runtime-apis",
    }
    sample = {"metadata": {"appworld": {"task_id": "task-1"}}}

    bootstrap = runtime.bootstrap(
        session=SimpleNamespace(),
        sample=sample,
        payload={"runtime_handle": runtime_handle},
    )
    runtime.save(sample=sample, payload={"runtime_handle": runtime_handle})

    assert calls[0][0] == "http://runtime-env"
    assert calls[0][2]["remote_apis_url"] == "http://runtime-apis"
    assert calls[1][0] == "http://runtime-env"
    assert calls[1][2]["task_id"] == "task-1"
    assert bootstrap["prompt_context"]["instruction"] == (
        "What is the title of the most-liked song in my Spotify playlists."
    )


@pytest.mark.fast
def test_appworld_runtime_uses_agentkit_v2_environment_profile_metadata() -> None:
    calls: list[tuple[str, str, dict[str, Any], int]] = []

    def requester(endpoint: str, method: str, payload: dict[str, Any], timeout_s: int) -> dict[str, Any]:
        calls.append((endpoint, method, payload, timeout_s))
        return {"output": {"method": method, "task_id": payload.get("task_id")}}

    session = SimpleNamespace(
        resource_lease=SimpleNamespace(
            metadata={
                "environment_profile": {
                    "metadata": {
                        "env_endpoint": "http://env-from-profile",
                        "apis_endpoint": "http://apis-from-profile",
                        "mcp_endpoint": "http://mcp-from-profile",
                    }
                }
            },
            handle_ref={},
        )
    )

    runtime = AppWorldRuntime(requester=requester)
    runtime.bootstrap(
        session=session,
        sample={"metadata": {"appworld": {"task_id": "task-1"}}},
        payload={},
    )

    assert calls[0][0] == "http://env-from-profile"
    assert calls[0][2]["remote_apis_url"] == "http://apis-from-profile"
    assert calls[0][2]["remote_mcp_url"] == "http://mcp-from-profile"


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


@pytest.mark.fast
def test_appworld_framework_loop_schema_injection_has_no_fixed_tmp_side_effects(monkeypatch) -> None:
    def fail_on_tmp_write(self: Path, *_args: Any, **_kwargs: Any) -> None:
        if str(self).startswith("/tmp/gage_inject_tool_schemas_"):
            raise AssertionError(f"unexpected fixed temp debug write: {self}")

    monkeypatch.setattr(Path, "write_text", fail_on_tmp_write)

    session = SimpleNamespace(
        prompt_context={
            "mcp_endpoint": "http://appworld-mcp",
            "allowed_apps": ["spotify"],
        }
    )
    sample = {"support_outputs": [{"mcp_client_id": "appworld_env"}]}
    live_schemas = [
        {
            "type": "function",
            "function": {"name": "spotify__search", "parameters": {"type": "object"}},
            "x-gage": {"mcp_client_id": "appworld_env"},
        }
    ]

    monkeypatch.setattr(framework_loop, "fetch_mcp_tool_schemas", lambda *_args, **_kwargs: live_schemas)
    assert framework_loop._inject_tool_schemas(session=session, sample=sample, payload={}) == live_schemas

    fallback_schemas = [{"type": "function", "function": {"name": "fallback", "parameters": {}}}]
    monkeypatch.setattr(framework_loop, "fetch_mcp_tool_schemas", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(framework_loop, "build_appworld_tools", lambda _sample: fallback_schemas)
    assert framework_loop._inject_tool_schemas(session=session, sample=sample, payload={}) == fallback_schemas
