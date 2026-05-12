from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricContext, MetricRegistry
from gage_eval.metrics.builtin.harbor import (
    ExternalTrialPassHatKMetric,
    HarborResolveRateMetric,
    HarborScoreMeanMetric,
)
from gage_eval.registry import import_asset_from_manifest, registry


@pytest.mark.fast
def test_harbor_resolve_rate_all_pass_returns_one() -> None:
    metric = HarborResolveRateMetric(_spec("harbor_resolve_rate"))

    result = metric.compute(_context(trial_results=[_trial(True, 1.0), _trial(True, 0.5)]))

    assert result.values["resolve_rate"] == 1.0
    assert result.metadata["completed_trial_count"] == 2


@pytest.mark.fast
def test_harbor_resolve_rate_half_pass_returns_half() -> None:
    metric = HarborResolveRateMetric(_spec("harbor_resolve_rate"))

    result = metric.compute(_context(trial_results=[_trial(True, 1.0), _trial(False, 0.0)]))

    assert result.values["resolve_rate"] == 0.5


@pytest.mark.fast
def test_harbor_score_mean_differs_from_resolve_rate_for_fractional_scores() -> None:
    context = _context(trial_results=[_trial(True, 0.9), _trial(False, 0.8)])

    resolve = HarborResolveRateMetric(_spec("harbor_resolve_rate")).compute(context)
    score = HarborScoreMeanMetric(_spec("harbor_score_mean")).compute(context)

    assert resolve.values["resolve_rate"] == 0.5
    assert score.values["score_mean"] == pytest.approx(0.85)


@pytest.mark.fast
def test_external_trial_pass_hat_k_counts_none_as_failed() -> None:
    metric = ExternalTrialPassHatKMetric(_spec("external_trial_pass_hat_k", params={"k": 1}))

    result = metric.compute(_context(pass_values=[True, None]))

    assert result.values["pass_hat"] == 0.5
    assert result.metadata["none_count"] == 1
    assert result.metadata["none_as_failed"] is True


@pytest.mark.fast
def test_external_trial_pass_hat_k_uses_combination_estimator() -> None:
    metric = ExternalTrialPassHatKMetric(_spec("external_trial_pass_hat_k", params={"k": 2}))

    result = metric.compute(_context(pass_values=[True, True, None]))

    assert result.values["pass_hat"] == pytest.approx(1 / 3)
    assert result.metadata["trial_count"] == 3
    assert result.metadata["success_count"] == 2


@pytest.mark.fast
def test_empty_completed_trials_return_zero_with_documented_semantics() -> None:
    resolve = HarborResolveRateMetric(_spec("harbor_resolve_rate")).compute(_context(trial_results=[]))
    score = HarborScoreMeanMetric(_spec("harbor_score_mean")).compute(_context(trial_results=[]))
    pass_hat = ExternalTrialPassHatKMetric(_spec("external_trial_pass_hat_k")).compute(_context(pass_values=[]))

    assert resolve.values["resolve_rate"] == 0.0
    assert resolve.metadata["empty_completed_trials_semantics"] == "0.0"
    assert score.values["score_mean"] == 0.0
    assert score.metadata["empty_completed_trials_semantics"] == "0.0"
    assert pass_hat.values["pass_hat"] == 0.0
    assert pass_hat.metadata["empty_trial_values_semantics"] == "0.0"


@pytest.mark.fast
def test_harbor_metrics_are_registered_from_metrics_manifest() -> None:
    import_asset_from_manifest("metrics", "harbor_resolve_rate", registry=registry)
    import_asset_from_manifest("metrics", "harbor_score_mean", registry=registry)
    import_asset_from_manifest("metrics", "external_trial_pass_hat_k", registry=registry)

    assert registry.get("metrics", "harbor_resolve_rate") is HarborResolveRateMetric
    assert registry.get("metrics", "harbor_score_mean") is HarborScoreMeanMetric
    assert registry.get("metrics", "external_trial_pass_hat_k") is ExternalTrialPassHatKMetric


@pytest.mark.fast
def test_metric_registry_builds_external_trial_pass_hat_k_with_mean_aggregation() -> None:
    instance = MetricRegistry().build_metric(
        _spec("external_trial_pass_hat_k", params={"k": 1}),
    )

    instance.evaluate(_context(sample_id="sample-1", pass_values=[True, None]))
    instance.evaluate(_context(sample_id="sample-2", pass_values=[False, False]))
    aggregated = instance.finalize()

    assert aggregated["aggregation"] == "mean"
    assert aggregated["values"]["pass_hat"] == pytest.approx(0.25)


def _spec(metric_id: str, *, params: dict | None = None) -> MetricSpec:
    return MetricSpec(metric_id=metric_id, implementation=metric_id, params=params or {})


def _context(
    *,
    sample_id: str = "sample-1",
    trial_results: list[dict] | None = None,
    pass_values: list[bool | None] | None = None,
) -> MetricContext:
    eval_result = {}
    if pass_values is not None:
        eval_result["external_trial_pass_values"] = pass_values
        eval_result["external_trial_metric_projection"] = {
            "trial_ids": [f"trial_{index:04d}" for index, _ in enumerate(pass_values, start=1)],
            "skipped_failed_trials": [
                {"trial_id": f"trial_{index:04d}"}
                for index, value in enumerate(pass_values, start=1)
                if value is None
            ],
        }
    return MetricContext(
        sample_id=sample_id,
        sample={"eval_result": eval_result, "trial_results": trial_results or []},
        model_output={},
        judge_output={"trial_results": trial_results or {}},
        args={},
        trace=object(),
    )


def _trial(passed: bool, score: float, *, status: str = "completed") -> dict:
    return {
        "trial_id": "trial_0001",
        "status": status,
        "verifier_result": {
            "passed": passed,
            "resolved": passed,
            "score": score,
            "reward": score,
        },
    }
