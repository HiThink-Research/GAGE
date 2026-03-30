from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricResult
from gage_eval.metrics.builtin.mme_aggregator import MMEAccPlusAggregator


def _make_result(
    sample_id: str,
    *,
    question_id: str | None,
    acc: float,
) -> MetricResult:
    metadata = {}
    if question_id is not None:
        metadata["question_id"] = question_id
    return MetricResult(
        sample_id=sample_id,
        values={"acc": acc},
        metadata=metadata,
    )


@pytest.mark.fast
def test_mme_acc_plus_aggregator_uses_compact_group_counters() -> None:
    aggregator = MMEAccPlusAggregator(
        MetricSpec(
            metric_id="mme_acc_plus",
            implementation="mme_acc_plus",
            aggregation="mme_acc_plus",
            params={},
        )
    )

    aggregator.add(_make_result("q1-0", question_id="q1", acc=1.0))
    aggregator.add(_make_result("q1-1", question_id="q1", acc=1.0))
    aggregator.add(_make_result("q2-0", question_id="q2", acc=1.0))
    aggregator.add(_make_result("q2-1", question_id="q2", acc=0.0))
    aggregator.add(_make_result("solo", question_id=None, acc=1.0))

    payload = aggregator.finalize().to_dict()

    assert payload["count"] == 5
    assert payload["values"]["acc"] == pytest.approx(0.8)
    assert payload["values"]["acc_plus"] == pytest.approx(0.5)
    assert payload["metadata"]["total_images"] == 2
    assert payload["metadata"]["correct_images"] == 1
    assert payload["metadata"]["group_count"] == 2
    assert payload["metadata"]["missing_group_id_samples"] == 1
