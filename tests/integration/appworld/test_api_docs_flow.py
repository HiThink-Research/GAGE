from __future__ import annotations

import asyncio

import pytest

from gage_eval.role.adapters.base import RoleAdapterState
from gage_eval.role.adapters.context_provider import ContextProviderAdapter


class ApiDocsStub:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def call_tool(self, name: str, arguments: dict) -> dict:
        self.calls.append({"name": name, "arguments": dict(arguments or {})})
        return {"tool": name, "arguments": dict(arguments or {})}


@pytest.mark.fast
def test_appworld_api_docs_context_provider_search() -> None:
    stub = ApiDocsStub()
    adapter = ContextProviderAdapter(
        adapter_id="api_docs_context",
        implementation="appworld_api_docs",
        mcp_client=stub,
        implementation_params={"max_pages": 1, "page_limit": 2, "max_chars": 2000},
    )

    payload = {"sample": {"instruction": "Find calendar APIs"}, "params": {"query": "calendar"}}
    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert result["api_docs_context"]
    assert stub.calls[0]["name"] == "api_docs__search_api_docs"
    assert result["observability_events"][0]["event"] == "api_docs_query"


@pytest.mark.fast
def test_appworld_api_docs_context_provider_caches_duplicate_calls() -> None:
    stub = ApiDocsStub()
    adapter = ContextProviderAdapter(
        adapter_id="api_docs_context",
        implementation="appworld_api_docs",
        mcp_client=stub,
        implementation_params={"max_pages": 1, "page_limit": 1, "max_chars": 2000},
    )

    payload = {"sample": {}, "params": {"api_names": {"spotify": ["login", "login"]}}}
    result = asyncio.run(adapter.ainvoke(payload, RoleAdapterState()))

    assert [call["name"] for call in stub.calls] == ["api_docs__show_api_doc"]
    events = result["observability_events"]
    assert len(events) == 2
    assert events[0]["payload"]["cache_hit"] is False
    assert events[1]["payload"]["cache_hit"] is True
