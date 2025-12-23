from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.observability.trace import ObservabilityTrace


class TupleMetric(SimpleMetric):
    def compute_value(self, context: MetricContext):
        return 0.7, {"info": "tuple-meta"}


class HookMetric(SimpleMetric):
    def compute_value(self, context: MetricContext):
        return 0.5

    def compute_metadata(self, context: MetricContext):
        return {"info": "hook-meta"}


def _ctx():
    return MetricContext(
        sample_id="s4",
        sample={},
        model_output={},
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )


def test_simple_metric_accepts_tuple_metadata():
    metric = TupleMetric(MetricSpec(metric_id="tuple", implementation="tuple_impl"))
    result = metric.compute(_ctx())
    assert result.values["score"] == 0.7
    assert result.metadata["info"] == "tuple-meta"


def test_simple_metric_metadata_hook():
    metric = HookMetric(MetricSpec(metric_id="hook", implementation="hook_impl"))
    result = metric.compute(_ctx())
    assert result.values["score"] == 0.5
    assert result.metadata["info"] == "hook-meta"
