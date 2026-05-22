from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReportContext


@pytest.mark.fast
def test_v1_renderer_accepts_unknown_optional_minor_fields() -> None:
    payload = ReportContext.minimal(run_id="run-demo").to_dict()
    payload["schema"]["minor"] = 1
    payload["new_optional_field"] = {"safe": True}

    errors = ReportContext.validate(payload, renderer_major=1)

    assert errors == []


@pytest.mark.fast
def test_missing_required_field_returns_diagnostic_error() -> None:
    payload = ReportContext.minimal(run_id="run-demo").to_dict()
    payload.pop("runtime_health")

    errors = ReportContext.validate(payload, renderer_major=1)

    assert any("runtime_health" in error for error in errors)


@pytest.mark.fast
def test_renderer_rejects_unknown_major_version() -> None:
    payload = ReportContext.minimal(run_id="run-demo").to_dict()
    payload["schema"]["major"] = 2

    errors = ReportContext.validate(payload, renderer_major=1)

    assert any("major" in error for error in errors)
