import pytest

from gage_eval.metrics.base import MetricContext
from gage_eval.metrics.utils import extract_field
from gage_eval.observability.trace import ObservabilityTrace


def _ctx(policy):
    return MetricContext(
        sample_id="s-missing",
        sample={"exists": {"value": 1}},
        model_output={},
        judge_output={},
        args={"on_missing_field": policy},
        trace=ObservabilityTrace(),
    )


def test_missing_field_ignore_returns_default():
    ctx = _ctx("ignore")
    assert extract_field(ctx, "sample.not_exist", default="fallback") == "fallback"


def test_missing_field_warn_logs(capsys):
    ctx = _ctx("warn")
    # WARN policy does not raise; it returns the default value.
    assert extract_field(ctx, "sample.not_exist", default=None) is None


def test_missing_field_error_raises():
    ctx = _ctx("error")
    with pytest.raises(KeyError):
        extract_field(ctx, "sample.not_exist", default=None)
