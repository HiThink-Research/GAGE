from __future__ import annotations

import pytest

from gage_eval.role.context.appworld_api_docs import AppWorldApiDocsContext


class FakeMcpClient:
    def __init__(self) -> None:
        self.list_calls = 0

    def list_tools(self) -> list[dict]:
        self.list_calls += 1
        return [
            {
                "name": "spotify__login",
                "description": "Login to Spotify",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "spotify__search_tracks",
                "description": "Search tracks",
                "inputSchema": {"type": "object"},
            },
            {
                "name": "supervisor__show_active_task",
                "description": "Show active task",
                "inputSchema": {"type": "object"},
            },
        ]


@pytest.mark.fast
def test_api_descriptions_context_uses_list_tools() -> None:
    client = FakeMcpClient()
    context = AppWorldApiDocsContext(
        mcp_client=client,
        mode="api_descriptions",
        include_app_descriptions=True,
        auto_discover_apps=True,
        max_chars=2000,
    )

    result = context.provide({"sample": {}, "params": {}}, state=None)
    text = result.get("api_descriptions_context")

    assert isinstance(text, str)
    assert "app_descriptions:" in text
    assert "spotify" in text
    assert "api_descriptions:" in text
    assert "login" in text
    assert "search_tracks" in text
    assert client.list_calls == 1
    assert "spotify.login" in result.get("api_descriptions_allowed_apis", [])
    assert "spotify__login" in result.get("api_descriptions_allowed_tools", [])
