from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.evaluation.cache import EvalCache
from gage_eval.metrics.aggregators import IdentityAggregator
from gage_eval.metrics.base import MetricResult
from gage_eval.metrics.runtime_context import AggregationRuntimeContext


def _make_result(sample_id: str, value: float) -> MetricResult:
    return MetricResult(
        sample_id=sample_id,
        values={"score": value},
        metadata={"sample_id": sample_id},
    )


@pytest.mark.fast
def test_identity_aggregator_uses_cache_ref_when_cache_available(tmp_path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="run-identity-cache")
    spec = MetricSpec(
        metric_id="identity_debug",
        implementation="exact_match",
        aggregation="identity",
        params={"preview_items": 1},
    )
    aggregator = IdentityAggregator(
        spec,
        runtime_context=AggregationRuntimeContext(
            run_id="run-identity-cache",
            task_id="t1",
            run_dir=cache.run_dir,
            cache_store=cache,
            details_namespace="task/t1",
        ),
    )

    aggregator.add(_make_result("s1", 1.0))
    aggregator.add(_make_result("s2", 0.0))

    payload = aggregator.finalize().to_dict()
    metadata = payload["metadata"]

    assert payload["values"] == {}
    assert metadata["storage_mode"] == "cache_ref"
    assert metadata["details_source"] == "eval_cache"
    assert metadata["details_file"] == str(cache.samples_jsonl)
    assert metadata["details_namespace"] == "task/t1"
    assert metadata["details_metric_id"] == "identity_debug"
    assert metadata["preview_count"] == 1
    assert metadata["preview_truncated"] is True
    assert len(metadata["samples_preview"]) == 1


@pytest.mark.fast
def test_identity_aggregator_keeps_inline_payload_for_small_runs() -> None:
    spec = MetricSpec(
        metric_id="identity_debug",
        implementation="exact_match",
        aggregation="identity",
        params={
            "detail_mode": "inline_preview",
            "preview_items": 2,
            "max_inline_items": 4,
        },
    )
    aggregator = IdentityAggregator(spec)

    aggregator.add(_make_result("s1", 1.0))
    aggregator.add(_make_result("s2", 0.0))

    payload = aggregator.finalize().to_dict()
    metadata = payload["metadata"]

    assert metadata["storage_mode"] == "inline"
    assert list(payload["values"]) == ["0", "1"]
    assert len(metadata["samples"]) == 2
    assert metadata["preview_count"] == 2
    assert metadata["preview_truncated"] is False


@pytest.mark.fast
def test_identity_aggregator_falls_back_to_preview_only_after_inline_limit() -> None:
    spec = MetricSpec(
        metric_id="identity_debug",
        implementation="exact_match",
        aggregation="identity",
        params={
            "detail_mode": "inline_preview",
            "preview_items": 1,
            "max_inline_items": 1,
        },
    )
    aggregator = IdentityAggregator(spec)

    aggregator.add(_make_result("s1", 1.0))
    aggregator.add(_make_result("s2", 0.0))

    payload = aggregator.finalize().to_dict()
    metadata = payload["metadata"]

    assert payload["values"] == {}
    assert metadata["storage_mode"] == "preview_only"
    assert metadata["switch_reason"] == "max_inline_items_exceeded"
    assert metadata["preview_count"] == 1
    assert metadata["preview_truncated"] is True
