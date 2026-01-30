from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.tau2 import (
    Tau2AgentCostMetric,
    Tau2PassMetric,
    Tau2RewardMetric,
    Tau2UserCostMetric,
)


def test_tau2_reward_and_pass_metrics(mock_trace) -> None:
    reward_spec = MetricSpec(metric_id="tau2_reward", implementation="tau2_reward", params={})
    pass_spec = MetricSpec(metric_id="tau2_pass", implementation="tau2_pass", params={})
    reward_metric = Tau2RewardMetric(reward_spec)
    pass_metric = Tau2PassMetric(pass_spec)
    context = MetricContext(
        sample_id="sample-1",
        sample={},
        model_output={},
        judge_output={"tau2": {"reward": 1.0}},
        args=reward_spec.params,
        trace=mock_trace,
    )

    reward_result = reward_metric.compute(context)
    pass_result = pass_metric.compute(context)

    assert reward_result.values["reward"] == 1.0
    assert pass_result.values["pass"] == 1.0


def test_tau2_cost_metrics(mock_trace) -> None:
    agent_spec = MetricSpec(metric_id="tau2_agent_cost", implementation="tau2_agent_cost", params={})
    user_spec = MetricSpec(metric_id="tau2_user_cost", implementation="tau2_user_cost", params={})
    agent_metric = Tau2AgentCostMetric(agent_spec)
    user_metric = Tau2UserCostMetric(user_spec)
    context = MetricContext(
        sample_id="sample-2",
        sample={},
        model_output={},
        judge_output={"tau2": {"agent_cost": 0.5, "user_cost": 0.25}},
        args=agent_spec.params,
        trace=mock_trace,
    )

    agent_result = agent_metric.compute(context)
    user_result = user_metric.compute(context)

    assert agent_result.values["agent_cost"] == 0.5
    assert user_result.values["user_cost"] == 0.25
