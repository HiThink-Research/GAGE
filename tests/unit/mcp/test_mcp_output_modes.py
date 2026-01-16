from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pytest
import yaml

from gage_eval.sandbox.integrations.appworld.mcp_client import normalize_mcp_call_result


class TextContentStub:
    def __init__(self, text: str) -> None:
        self.text = text


class ResultStub:
    def __init__(
        self,
        *,
        structured_content: Optional[Dict[str, Any]] = None,
        content: Optional[List[Any]] = None,
        is_error: bool = False,
    ) -> None:
        self.structuredContent = structured_content
        self.content = content or []
        self.isError = is_error


@pytest.mark.fast
def test_streamable_mcp_output_parsing() -> None:
    structured = ResultStub(structured_content={"response": {"message": "ok"}})
    assert normalize_mcp_call_result(structured)["response"]["message"] == "ok"

    content_only = ResultStub(content=[TextContentStub('{"response": {"message": "done"}}')])
    assert normalize_mcp_call_result(content_only)["response"]["message"] == "done"

    plain_text = ResultStub(content=[TextContentStub("bad request")], is_error=True)
    parsed = normalize_mcp_call_result(plain_text)
    assert parsed["response"]["message"] == "bad request"
    assert parsed["response"]["is_error"] is True


@pytest.mark.io
def test_appworld_output_type_configured() -> None:
    config_path = Path(__file__).resolve().parents[3] / "config" / "custom" / "appworld_agent_demo.yaml"
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    mcp_client = payload["mcp_clients"][0]

    assert mcp_client["params"]["output_type"] == "structured_data_only"
