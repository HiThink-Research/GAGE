from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.failure_clusterer import FailureClusterer
from gage_eval.reporting.contracts import AttentionCase, Severity


@pytest.mark.fast
def test_failure_clusterer_separates_runtime_and_system_reason_counts() -> None:
    case = AttentionCase(
        case_id="task/sample",
        severity=Severity.HIGH,
        scoring={"frequency": 1, "impact": "high", "actionability": "high", "priority_score": 0.9},
        reason_codes=["scheduler.failed", "system.report_context.incomplete"],
        summary="Failed",
    )

    result = FailureClusterer().cluster([case])

    assert result.reason_code_counts["runtime"]["scheduler.failed"] == 1
    assert result.reason_code_counts["system"]["system.report_context.incomplete"] == 1
    assert result.failure_clusters[0].cluster_key == ["scheduler.failed"]
