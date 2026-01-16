from __future__ import annotations

import pytest

from gage_eval.tools.appworld_tool_docs_probe import build_probe_report


@pytest.mark.fast
def test_tool_docs_probe_report() -> None:
    tools = [
        {
            "name": "spotify__search",
            "description": "Search Spotify",
            "inputSchema": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    report = build_probe_report(tools, max_chars=500)

    assert report["apps"] == 1
    assert report["endpoints"] == 1
    assert "apps_detail" in report
