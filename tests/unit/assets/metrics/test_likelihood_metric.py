import math

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.builtin.likelihood import LikelihoodMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


def _ctx(model_output=None):
    return MetricContext(
        sample_id="s1",
        sample={},
        model_output=model_output or {},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )


def test_likelihood_uses_loss_when_available():
    spec = MetricSpec(metric_id="llh", implementation="likelihood", params={})
    metric = LikelihoodMetric(spec)
    result = metric.compute(_ctx({"loss": 1.2}))
    assert result.values["nll"] == 1.2
    assert result.metadata["source"] == "loss"


def test_likelihood_from_token_logprobs():
    spec = MetricSpec(metric_id="llh", implementation="likelihood", params={})
    metric = LikelihoodMetric(spec)
    # avg logprob = ( -1 + -2 ) / 2 = -1.5 => nll = 1.5
    result = metric.compute(_ctx({"token_logprobs": [-1.0, -2.0]}))
    assert result.values["nll"] == 1.5
    assert result.metadata["source"] == "token_logprobs"
    assert result.metadata["token_count"] == 2


def test_likelihood_ppl_mode():
    spec = MetricSpec(metric_id="llh", implementation="likelihood", params={"metric_type": "ppl"})
    metric = LikelihoodMetric(spec)
    result = metric.compute(_ctx({"loss": 2.0}))
    assert math.isclose(result.values["nll"], math.exp(2.0))
    assert result.metadata["metric_type"] == "ppl"


def test_likelihood_missing_returns_error_metadata():
    spec = MetricSpec(metric_id="llh", implementation="likelihood", params={})
    metric = LikelihoodMetric(spec)
    result = metric.compute(_ctx({}))
    assert result.values["nll"] == 0.0
    assert result.metadata["error"] == "missing_loss_or_logprobs"
