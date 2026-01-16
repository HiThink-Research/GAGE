from __future__ import annotations

from typing import Any, Dict, List

import pytest

from gage_eval.sandbox.integrations.appworld import mcp_client as appworld_mcp_client
from gage_eval.sandbox.integrations.appworld.mcp_client import AppWorldStreamableMcpClient


@pytest.mark.fast
def test_streamable_mcp_client_static_tools_and_allowlist() -> None:
    tools = [
        {"name": "allowed_tool", "description": "Allowed", "inputSchema": {}},
        {"name": "blocked_tool", "description": "Blocked", "inputSchema": {}},
    ]
    client = AppWorldStreamableMcpClient(
        mcp_client_id="appworld_env",
        endpoint="http://127.0.0.1:5001",
        allowlist=["allowed_tool"],
        params={"tools": tools},
    )

    resolved = client.list_tools()
    names = [tool.get("name") for tool in resolved]
    assert names == ["allowed_tool"]


@pytest.mark.fast
def test_streamable_mcp_client_requester_roundtrip() -> None:
    calls: List[tuple[str, Dict[str, Any]]] = []

    def requester(method: str, payload: Dict[str, Any]) -> Any:
        calls.append((method, payload))
        if method == "list_tools":
            return [{"name": "step", "description": "step", "inputSchema": {}}]
        if method == "call_tool":
            return {"response": {"status": "ok"}}
        return {}

    client = AppWorldStreamableMcpClient(
        mcp_client_id="appworld_env",
        endpoint="http://127.0.0.1:5001",
        params={"requester": requester},
    )
    tools = client.list_tools()
    result = client.call_tool("step", {"action": "noop"})

    assert tools[0]["name"] == "step"
    assert result["response"]["status"] == "ok"
    assert calls[0][0] == "list_tools"
    assert calls[1][0] == "call_tool"


@pytest.mark.fast
def test_streamable_mcp_client_session_retry(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AppWorldStreamableMcpClient(
        mcp_client_id="appworld_env",
        endpoint="http://127.0.0.1:5001",
        params={"session_retry_attempts": 2, "session_retry_delay_s": 0},
    )
    calls = {"count": 0}

    def fake_once() -> None:
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("boom")
        client._session = object()

    monkeypatch.setattr(client, "_ensure_session_once", fake_once)

    client._ensure_session()

    assert calls["count"] == 2


@pytest.mark.fast
def test_streamable_mcp_client_session_retry_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    client = AppWorldStreamableMcpClient(
        mcp_client_id="appworld_env",
        endpoint="http://127.0.0.1:5001",
        params={"session_retry_timeout_s": 0.25, "session_retry_delay_s": 0.1},
    )
    calls = {"count": 0}
    clock = {"now": 0.0}

    def fake_once() -> None:
        calls["count"] += 1
        raise RuntimeError("boom")

    def fake_monotonic() -> float:
        return clock["now"]

    def fake_sleep(delay: float) -> None:
        clock["now"] += delay

    monkeypatch.setattr(client, "_ensure_session_once", fake_once)
    monkeypatch.setattr(appworld_mcp_client.time, "monotonic", fake_monotonic)
    monkeypatch.setattr(appworld_mcp_client.time, "sleep", fake_sleep)

    with pytest.raises(RuntimeError, match="boom"):
        client._ensure_session()

    assert calls["count"] == 4
