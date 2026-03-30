from __future__ import annotations

import json

import pytest

from gage_eval.evaluation.cache import EvalCache


@pytest.mark.fast
def test_merge_summary_fields_patches_existing_summary(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="summary-patch-existing")
    cache.write_summary({"sample_count": 1})

    cache.merge_summary_fields(
        {
            "observability_closed_cleanly": True,
            "observability_close_mode": "drain",
        }
    )

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["sample_count"] == 1
    assert summary["observability_closed_cleanly"] is True
    assert summary["observability_close_mode"] == "drain"


@pytest.mark.fast
def test_merge_summary_fields_defers_until_summary_exists(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="summary-patch-deferred")

    cache.merge_summary_fields(
        {
            "observability_close_mode": "best_effort",
            "observability_close_warning": "best_effort mode may leave observability data incomplete",
        }
    )
    cache.write_summary({"sample_count": 2})

    summary = json.loads((cache.run_dir / "summary.json").read_text(encoding="utf-8"))

    assert summary["sample_count"] == 2
    assert summary["observability_close_mode"] == "best_effort"
    assert "observability_close_warning" in summary
