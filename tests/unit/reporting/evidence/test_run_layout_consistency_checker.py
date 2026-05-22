from __future__ import annotations

import json

import pytest

from gage_eval.reporting.evidence.consistency_checker import RunLayoutConsistencyChecker


@pytest.mark.io
def test_checker_reports_sample_count_mismatch(tmp_path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "summary.json").write_text(json.dumps({"sample_count": 2}), encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(json.dumps({"sample_id": "one"}) + "\n", encoding="utf-8")

    diagnostics = RunLayoutConsistencyChecker().check(run_dir)

    assert any(item["code"] == "report_pack.sample_count_mismatch" for item in diagnostics.warnings)


@pytest.mark.io
def test_checker_marks_namespace_sample_json_as_derived_detail(tmp_path) -> None:
    run_dir = tmp_path / "run"
    detail = run_dir / "samples" / "task" / "sample-1.json"
    detail.parent.mkdir(parents=True)
    detail.write_text("{}", encoding="utf-8")
    (run_dir / "samples.jsonl").write_text(json.dumps({"sample_id": "sample-1"}) + "\n", encoding="utf-8")

    diagnostics = RunLayoutConsistencyChecker().check(run_dir)

    assert diagnostics.derived_detail_count == 1
