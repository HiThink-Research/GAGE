from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.outlier_detector import OutlierDetector


@pytest.mark.fast
def test_outlier_detector_keeps_bounded_top_k() -> None:
    records = [{"sample_id": f"s{i}", "latency_s": i, "total_tokens": i * 10} for i in range(20)]

    outliers = OutlierDetector(top_k=3).detect(records)

    latency = next(group for group in outliers if group.metric_id == "latency_s")
    assert [item["sample_id"] for item in latency.top_k] == ["s19", "s18", "s17"]
