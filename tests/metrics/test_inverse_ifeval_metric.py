from __future__ import annotations

import pytest

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.inverse_ifeval import (
    InverseIFEvalJudgePassRateMetric,
    InverseIFEvalPassRateMetric,
)


@pytest.mark.fast
def test_inverse_ifeval_metric_explicit_constraints_path() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_pass_rate",
        implementation="inverse_ifeval_pass_rate",
        params={},
    )
    metric = InverseIFEvalPassRateMetric(spec)
    context = MetricContext(
        sample_id="s1",
        sample={
            "metadata": {
                "constraints": [
                    {"id": "r1", "type": "must_contain", "value": "hello"},
                    {"id": "r2", "type": "must_not_contain", "value": "forbidden"},
                ]
            }
        },
        model_output={"answer": "Hello world"},
        judge_output={},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 1.0
    assert result.values["passed"] == 1.0
    assert result.metadata["total_rules"] == 2
    assert result.metadata["failed_rule_ids"] == []


@pytest.mark.fast
def test_inverse_ifeval_metric_instruction_id_path() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_pass_rate",
        implementation="inverse_ifeval_pass_rate",
        params={},
    )
    metric = InverseIFEvalPassRateMetric(spec)
    context = MetricContext(
        sample_id="s2",
        sample={
            "metadata": {
                "instruction_id_list": ["contains", "max_length"],
                "kwargs": {"contains": "ok", "max_length": 20},
            }
        },
        model_output={"answer": "ok"},
        judge_output={},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 1.0
    assert result.values["passed"] == 1.0
    assert result.metadata["rule_origin"] == "instruction_id_list"


@pytest.mark.fast
def test_inverse_ifeval_metric_unsupported_rule_is_failed() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_pass_rate",
        implementation="inverse_ifeval_pass_rate",
        params={},
    )
    metric = InverseIFEvalPassRateMetric(spec)
    context = MetricContext(
        sample_id="s3",
        sample={
            "metadata": {
                "constraints": [
                    {"id": "u1", "type": "some_unsupported_rule"},
                ]
            }
        },
        model_output={"answer": "anything"},
        judge_output={},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 0.0
    assert result.values["passed"] == 0.0
    assert result.metadata["unsupported_rule_ids"] == ["u1"]
    assert result.metadata["failed_rule_ids"] == ["u1"]


@pytest.mark.fast
def test_inverse_ifeval_metric_empty_or_malformed_constraints() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_pass_rate",
        implementation="inverse_ifeval_pass_rate",
        params={},
    )
    metric = InverseIFEvalPassRateMetric(spec)
    context = MetricContext(
        sample_id="s4",
        sample={
            "metadata": {
                "constraints": [],
                "instruction_id_list": [None, "", "not_supported_id"],
            }
        },
        model_output={"answer": "text"},
        judge_output={},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 0.0
    assert result.values["passed"] == 0.0
    assert result.metadata["total_rules"] == 1
    assert result.metadata["unsupported_rule_ids"] == ["inst_2_not_supported_id"]


@pytest.mark.fast
def test_inverse_ifeval_judge_metric_accepts_numeric_answer() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_judge_pass_rate",
        implementation="inverse_ifeval_judge_pass_rate",
        params={"threshold": 0.5},
    )
    metric = InverseIFEvalJudgePassRateMetric(spec)
    context = MetricContext(
        sample_id="j1",
        sample={},
        model_output={},
        judge_output={"answer": "1"},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 1.0
    assert result.values["passed"] == 1.0
    assert result.metadata["judge_source"] == "judge_output.answer"


@pytest.mark.fast
def test_inverse_ifeval_judge_metric_accepts_verdict_tokens() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_judge_pass_rate",
        implementation="inverse_ifeval_judge_pass_rate",
        params={"threshold": 0.5},
    )
    metric = InverseIFEvalJudgePassRateMetric(spec)
    context = MetricContext(
        sample_id="j2",
        sample={},
        model_output={},
        judge_output={"verdict": "FAILED"},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 0.0
    assert result.values["passed"] == 0.0
    assert result.metadata["judge_source"] == "judge_output.verdict"


@pytest.mark.fast
def test_inverse_ifeval_judge_metric_accepts_json_answer() -> None:
    spec = MetricSpec(
        metric_id="inverse_ifeval_judge_pass_rate",
        implementation="inverse_ifeval_judge_pass_rate",
        params={"threshold": 0.5},
    )
    metric = InverseIFEvalJudgePassRateMetric(spec)
    context = MetricContext(
        sample_id="j3",
        sample={},
        model_output={},
        judge_output={"answer": "{\"score\": 0.83, \"verdict\": \"pass\"}"},
        args={},
        trace=None,
    )

    result = metric.compute(context)

    assert result.values["pass_rate"] == 1.0
    assert result.metadata["judge_score"] == pytest.approx(0.83)
