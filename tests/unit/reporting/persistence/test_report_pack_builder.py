from __future__ import annotations

import json

import pytest

from gage_eval.reporting.contracts import ReportContext
from gage_eval.reporting.persistence.pack_builder import ReportPackBuilder


class _LeakyRenderer:
    def render(self, payload: dict) -> str:
        return "Authorization: Bearer abc123 alice@example.com"


@pytest.mark.io
def test_report_pack_builder_writes_core_files(tmp_path) -> None:
    context = ReportContext.minimal(run_id="run")

    result = ReportPackBuilder().write(tmp_path, context)

    pack = tmp_path / "report_pack"
    assert result["report_pack_status"] == "completed"
    assert (pack / "report_context.json").exists()
    assert (pack / "report_context.md").exists()
    assert (pack / "report.html").exists()
    assert (pack / "prompt.txt").exists()
    assert (pack / "assets_manifest.json").exists()
    assert json.loads((pack / "diagnostics.json").read_text(encoding="utf-8"))["report_pack_status"] == "completed"


@pytest.mark.io
def test_report_pack_builder_redacts_renderer_output_before_write(tmp_path) -> None:
    context = ReportContext.minimal(run_id="run")
    builder = ReportPackBuilder(
        markdown_renderer=_LeakyRenderer(),
        html_renderer=_LeakyRenderer(),
        prompt_renderer=_LeakyRenderer(),
    )

    builder.write(tmp_path, context)

    pack = tmp_path / "report_pack"
    for name in ("report_context.md", "report.html", "prompt.txt"):
        text = (pack / name).read_text(encoding="utf-8")
        assert "Bearer abc123" not in text
        assert "alice@example.com" not in text
        assert "<redacted:" in text

    diagnostics = json.loads((pack / "diagnostics.json").read_text(encoding="utf-8"))
    assert any(
        warning.get("code") == "report_pack.secret_redacted"
        for warning in diagnostics["warnings"]
    )


@pytest.mark.io
def test_report_pack_builder_keeps_context_json_parseable_after_redaction(tmp_path) -> None:
    context = ReportContext.minimal(run_id="run").to_dict()
    context["case_details"] = {
        "case-1": {
            "tool_call_summary": {
                "input": {
                    "username": "user@example.com",
                    "password": "password123",
                }
            }
        }
    }

    ReportPackBuilder().write(tmp_path, context)

    text = (tmp_path / "report_pack" / "report_context.json").read_text(encoding="utf-8")
    payload = json.loads(text)
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "user@example.com" not in serialized
    assert "password123" not in serialized
    assert payload["case_details"]["case-1"]["tool_call_summary"]["input"]["username"] == "<redacted:email>"
    assert payload["case_details"]["case-1"]["tool_call_summary"]["input"]["password"] == "<redacted:secret>"
