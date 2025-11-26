"""Metric核心抽象：上下文、结果与基类定义。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, TYPE_CHECKING

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec

if TYPE_CHECKING:  # 避免循环导入
    from gage_eval.observability.trace import ObservabilityTrace


@dataclass(frozen=True)
class MetricContext:
    """传入 Metric 的运行期上下文。"""

    sample_id: str
    sample: Mapping[str, Any]
    model_output: Mapping[str, Any]
    judge_output: Mapping[str, Any]
    args: Mapping[str, Any]
    trace: "ObservabilityTrace"


@dataclass(frozen=True)
class MetricResult:
    """每个样本的 Metric 输出。"""

    sample_id: str
    values: Dict[str, float]
    metadata: Dict[str, Any] = field(default_factory=dict)
    explanation: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"sample_id": self.sample_id, "values": self.values}
        if self.metadata:
            payload["metadata"] = self.metadata
        if self.explanation:
            payload["explanation"] = self.explanation
        return payload


@dataclass(frozen=True)
class AggregatedMetric:
    """聚合后的 Metric 结果。"""

    metric_id: str
    aggregation: str
    values: Dict[str, float]
    count: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        payload = {
            "metric_id": self.metric_id,
            "aggregation": self.aggregation,
            "values": self.values,
            "count": self.count,
        }
        if self.metadata:
            payload["metadata"] = self.metadata
        return payload


class BaseMetric:
    """所有 Metric 的抽象基类。"""

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec
        self.args = dict(spec.params)
        self.setup()
        logger.debug("Metric '{}' initialized", spec.metric_id)

    def setup(self) -> None:
        """可选的初始化钩子，适合加载模型等重资源。"""

    def teardown(self) -> None:
        """可选的清理钩子。"""
        logger.debug("Metric '{}' teardown invoked", self.spec.metric_id)

    def compute(self, context: MetricContext) -> MetricResult:  # pragma: no cover
        raise NotImplementedError


class SimpleMetric(BaseMetric):
    """仅返回单一浮点值的 Metric 便捷基类。"""

    value_key: str = "score"

    def compute_value(self, context: MetricContext) -> float:  # pragma: no cover
        raise NotImplementedError

    def compute(self, context: MetricContext) -> MetricResult:
        value = float(self.compute_value(context))
        logger.trace("Metric '{}' computed value={}", self.spec.metric_id, value)
        return MetricResult(sample_id=context.sample_id, values={self.value_key: value})
