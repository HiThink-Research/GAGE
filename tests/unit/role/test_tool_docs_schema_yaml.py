from __future__ import annotations

import pytest

from gage_eval.role.toolchain.tool_docs import build_app_catalog, build_tool_documentation


@pytest.mark.fast
def test_schema_yaml_documentation_includes_response_schema() -> None:
    long_description = "A" * 200
    tools = [
        {
            "name": "calendar__create_event",
            "description": long_description,
            "inputSchema": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "visibility": {"type": "string", "enum": ["public", "private"]},
                },
                "required": ["title"],
            },
            "outputSchema": {
                "type": "object",
                "properties": {"event_id": {"type": "string"}},
                "required": ["event_id"],
            },
        }
    ]

    catalog = build_app_catalog(tools)
    documentation = build_tool_documentation(catalog, doc_format="schema_yaml")
    text = documentation.text

    assert documentation.meta["doc_format"] == "schema_yaml"
    assert "calendar:" in text
    assert "create_event:" in text
    assert "parameters:" in text
    assert "response_schema:" in text
    assert '"event_id"' in text

    description_line = next(line for line in text.splitlines() if "description:" in line)
    assert description_line.endswith("...")
