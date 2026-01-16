from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.appworld import (
    AppWorldDifficultyMetric,
    AppWorldFailCountMetric,
    AppWorldPassCountMetric,
    AppWorldTGCMetric,
)


def test_appworld_tgc_metric(mock_trace) -> None:
    spec = MetricSpec(metric_id="appworld_tgc", implementation="appworld_tgc", params={})
    metric = AppWorldTGCMetric(spec)
    context = MetricContext(
        sample_id="sample-1",
        sample={},
        model_output={},
        judge_output={"appworld": {"tgc": 0.75}},
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.values["tgc"] == 0.75


def test_appworld_pass_fail_metrics(mock_trace) -> None:
    pass_spec = MetricSpec(metric_id="appworld_pass_count", implementation="appworld_pass_count", params={})
    fail_spec = MetricSpec(metric_id="appworld_fail_count", implementation="appworld_fail_count", params={})
    pass_metric = AppWorldPassCountMetric(pass_spec)
    fail_metric = AppWorldFailCountMetric(fail_spec)
    context = MetricContext(
        sample_id="sample-2",
        sample={},
        model_output={},
        judge_output={"appworld": {"tests": {"passes": ["a", "b"], "fails": ["c"]}}},
        args=pass_spec.params,
        trace=mock_trace,
    )

    pass_result = pass_metric.compute(context)
    fail_result = fail_metric.compute(context)

    assert pass_result.values["passes"] == 2.0
    assert fail_result.values["fails"] == 1.0


def test_appworld_difficulty_metric(mock_trace) -> None:
    spec = MetricSpec(
        metric_id="appworld_difficulty",
        implementation="appworld_difficulty",
        aggregation="categorical_count",
        params={"category_field": "difficulty"},
    )
    metric = AppWorldDifficultyMetric(spec)
    context = MetricContext(
        sample_id="sample-3",
        sample={},
        model_output={},
        judge_output={"appworld": {"difficulty": "hard"}},
        args=spec.params,
        trace=mock_trace,
    )

    result = metric.compute(context)

    assert result.metadata["difficulty"] == "hard"
