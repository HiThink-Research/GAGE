from __future__ import annotations

from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics import MetricContext, MetricRegistry
from gage_eval.observability.trace import ObservabilityTrace
import gage_eval.metrics.builtin.gomoku  # noqa: F401


def _ctx(model_output):
    return MetricContext(
        sample_id="s1",
        sample={},
        model_output=model_output,
        judge_output={},
        args={},
        trace=ObservabilityTrace(),
    )


def test_gomoku_win_rate_metric():
    registry = MetricRegistry()
    metric = registry.build_metric(
        MetricSpec(
            metric_id="gomoku_win_rate",
            implementation="gomoku_win_rate",
            params={"target_player": "Black"},
        )
    )

    win = metric.evaluate(_ctx({"winner": "Black"}))
    loss = metric.evaluate(_ctx({"winner": "White"}))

    assert win.values["win"] == 1.0
    assert loss.values["win"] == 0.0


def test_gomoku_illegal_rate_metric():
    registry = MetricRegistry()
    metric = registry.build_metric(
        MetricSpec(metric_id="gomoku_illegal_rate", implementation="gomoku_illegal_rate")
    )

    legal = metric.evaluate(_ctx({"illegal_move_count": 0}))
    illegal = metric.evaluate(_ctx({"illegal_move_count": 2}))

    assert legal.values["illegal"] == 0.0
    assert illegal.values["illegal"] == 1.0


def test_gomoku_average_turns_metric():
    registry = MetricRegistry()
    metric = registry.build_metric(
        MetricSpec(metric_id="gomoku_avg_turns", implementation="gomoku_avg_turns")
    )

    result = metric.evaluate(_ctx({"move_count": 12}))

    assert result.values["turns"] == 12.0
