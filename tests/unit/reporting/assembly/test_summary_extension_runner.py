from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.extension_runner import SummaryExtensionRunner
from gage_eval.reporting.contracts import ReportContext, SummaryGeneratorResult


class _DuplicateSectionGenerator:
    name = "demo"

    def generate(self, context):
        return SummaryGeneratorResult(
            generator_id="demo",
            summary_sections=[
                {"section_id": "overview", "title": "One", "severity": "info"},
                {"section_id": "overview", "title": "Two", "severity": "info"},
            ],
            legacy_payload={"demo_summary": {"ok": True}},
        )


@pytest.mark.fast
def test_summary_extension_runner_canonicalizes_section_ids_and_warns_on_duplicate() -> None:
    result = SummaryExtensionRunner().run([_DuplicateSectionGenerator()], ReportContext.minimal("run").to_dict())

    assert result.summary_sections[0]["section_id"] == "demo/overview"
    assert len(result.summary_sections) == 1
    assert result.legacy_payload["demo_summary"]["ok"] is True
    assert any(item["code"] == "report_pack.section_id_duplicate" for item in result.diagnostics["warnings"])
