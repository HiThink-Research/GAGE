from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.metric_collector import MetricSummaryCollector


@pytest.mark.fast
def test_metric_summary_collector_formats_values_and_preserves_raw() -> None:
    metrics = [{"metric_id": "reward", "scope": "run", "values": {"mean": 0.123456, "label": "ok"}}]

    formatted = MetricSummaryCollector().collect(metrics)

    assert formatted[0]["values"]["mean"] == "0.12346"
    assert formatted[0]["values"]["label"] == "ok"
    assert formatted[0]["raw_values"]["mean"] == 0.123456
