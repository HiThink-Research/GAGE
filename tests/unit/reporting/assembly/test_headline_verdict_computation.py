from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.reporting.assembly.headline_builder import HeadlineBuilder


@pytest.mark.fast
@pytest.mark.parametrize(
    "runtime_health,diagnostics,expected",
    [
        ({"completed_count": 1, "failed_count": 0}, {"report_pack_status": "completed"}, "passed"),
        ({"completed_count": 1, "failed_count": 1}, {"report_pack_status": "completed"}, "passed_with_warnings"),
        ({"completed_count": 0, "failed_count": 1}, {"report_pack_status": "completed"}, "failed"),
        ({"completed_count": 0, "failed_count": 0, "aborted_count": 1}, {"report_pack_status": "completed"}, "failed"),
        ({"completed_count": 1, "failed_count": 0}, {"report_pack_status": "degraded"}, "degraded"),
    ],
)
def test_headline_verdict_is_deterministic(runtime_health: dict, diagnostics: dict, expected: str) -> None:
    headline = HeadlineBuilder().build(
        metrics=[],
        runtime_health=runtime_health,
        attention_cases=[],
        outliers=[],
        failure_clusters=[],
        diagnostics=diagnostics,
    )

    assert headline["verdict"] == expected
    assert headline["one_line_summary"]


@pytest.mark.fast
def test_headline_builder_outputs_design_contract_fields() -> None:
    headline = HeadlineBuilder().build(
        metrics=[
            {"metric_id": "reward_mean", "primary": True},
            {"metric_id": "accuracy"},
            {"metric_id": "latency_s"},
            {"metric_id": "total_tokens"},
        ],
        runtime_health={
            "sample_count": 10,
            "completed_count": 8,
            "failed_count": 1,
            "aborted_count": 1,
            "scheduler_failed_count": 1,
        },
        attention_cases=[
            SimpleNamespace(case_id="case-1"),
            {"case_id": "case-2"},
            {"case_id": "case-3"},
            {"case_id": "case-4"},
        ],
        outliers=[
            SimpleNamespace(metric_id="latency_s"),
            {"metric_id": "total_tokens"},
            {"metric_id": "cost_usd"},
            {"metric_id": "extra"},
        ],
        failure_clusters=[
            SimpleNamespace(cluster_id="cluster-1"),
            {"cluster_id": "cluster-2"},
            {"cluster_id": "cluster-3"},
            {"cluster_id": "cluster-4"},
        ],
        diagnostics={"report_pack_status": "completed"},
    )

    assert headline["verdict_reason"] == "8/10 samples completed; 1 failed; 1 aborted; 1 scheduler failure"
    assert headline["key_metric_ids"] == ["reward_mean", "accuracy", "latency_s"]
    assert headline["top_attention_case_ids"] == ["case-1", "case-2", "case-3"]
    assert headline["top_failure_cluster_ids"] == ["cluster-1", "cluster-2", "cluster-3"]
    assert headline["top_outlier_metric_ids"] == ["latency_s", "total_tokens", "cost_usd"]
