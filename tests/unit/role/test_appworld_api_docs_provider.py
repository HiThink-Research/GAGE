from __future__ import annotations

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.context.appworld_api_docs import AppWorldApiDocsContext


class ApiDocsStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append({"name": name, "arguments": dict(arguments or {})})
        return {"tool": name, "arguments": dict(arguments or {})}


@pytest.mark.fast
def test_appworld_api_docs_context_search_query() -> None:
    stub = ApiDocsStub()
    provider = AppWorldApiDocsContext(mcp_client=stub, max_pages=1, page_limit=2, max_chars=2000)
    payload = {
        "sample": {"instruction": "Find calendar APIs"},
        "params": {"query": "calendar"},
    }

    result = provider.provide(payload, RoleAdapterState())

    assert stub.calls
    assert stub.calls[0]["name"] == "api_docs__search_api_docs"
    assert stub.calls[0]["arguments"]["page_index"] == 0
    assert "# search_page_0" in result["api_docs_context"]
    assert result["observability_events"][0]["event"] == "api_docs_query"
    assert result["observability_events"][0]["payload"]["cache_hit"] is False


@pytest.mark.fast
def test_appworld_api_docs_context_caches_duplicate_calls() -> None:
    stub = ApiDocsStub()
    provider = AppWorldApiDocsContext(mcp_client=stub, max_pages=1, page_limit=1, max_chars=2000)
    payload = {
        "sample": {},
        "params": {"api_names": {"spotify": ["login", "login"]}},
    }

    result = provider.provide(payload, RoleAdapterState())

    assert [call["name"] for call in stub.calls] == ["api_docs__show_api_doc"]
    events = result["observability_events"]
    assert len(events) == 2
    assert events[0]["payload"]["cache_hit"] is False
    assert events[1]["payload"]["cache_hit"] is True
