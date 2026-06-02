from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext, SummaryGeneratorResult
from gage_eval.reporting.summary_generators.base import SummaryGenerator


class _Generator(SummaryGenerator):
    name = "demo_summary"

    def generate(self, context):
        return SummaryGeneratorResult(
            generator_id=self.name,
            summary_sections=[{"section_id": "overview", "title": "Demo", "severity": "info"}],
            legacy_payload={"demo_summary": {"sample_count": len(context.get("samples", []))}},
        )


@pytest.mark.fast
def test_summary_generator_v2_uses_context_only() -> None:
    result = _Generator().generate(ReportContext.minimal("run").to_dict())

    assert result.generator_id == "demo_summary"
    assert result.summary_sections[0]["section_id"] == "overview"
    assert result.legacy_payload["demo_summary"]["sample_count"] == 0


@pytest.mark.fast
def test_base_summary_generator_rejects_old_cache_argument_name() -> None:
    assert SummaryGenerator.contract_version == "gage.summary_generator.v2"
