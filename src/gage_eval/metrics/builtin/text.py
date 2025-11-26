"""文本/通用类型内置指标实现。"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from gage_eval.metrics.base import MetricContext, SimpleMetric
from gage_eval.registry import registry


def _walk_mapping(source: Any, path: Optional[str], default: Any = None) -> Any:
    if source is None or path is None:
        return default
    current = source
    for segment in path.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
        else:
            return default
    return current


def _extract_field(context: MetricContext, descriptor: Optional[str], default: Any = None) -> Any:
    if not descriptor:
        return default
    roots = {
        "sample": context.sample,
        "model_output": context.model_output,
        "judge_output": context.judge_output,
    }
    parts = descriptor.split(".")
    root_key = parts[0]
    if root_key in roots:
        base = roots[root_key]
        tail = ".".join(parts[1:]) if len(parts) > 1 else None
        return _walk_mapping(base, tail, default=default)
    return _walk_mapping(context.sample, descriptor, default=default)


def _normalize_text(value: Any, *, case_sensitive: bool, strip: bool) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    if strip:
        text = text.strip()
    if not case_sensitive:
        text = text.lower()
    return text


@registry.asset(
    "metrics",
    "exact_match",
    desc="严格文本匹配指标",
    tags=("text",),
    default_aggregation="mean",
)
class ExactMatchMetric(SimpleMetric):
    """严格匹配模型输出与参考答案。"""

    def compute_value(self, context: MetricContext) -> float:
        label_field = self.args.get("label_field", "label")
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        case_sensitive = bool(self.args.get("case_sensitive", False))
        strip = bool(self.args.get("strip_whitespace", True))

        reference = _extract_field(context, label_field)
        prediction = _extract_field(context, prediction_field, default="")

        normalized_ref = _normalize_text(reference, case_sensitive=case_sensitive, strip=strip)
        normalized_pred = _normalize_text(prediction, case_sensitive=case_sensitive, strip=strip)
        if normalized_ref is None or normalized_pred is None:
            return 0.0
        return 1.0 if normalized_ref == normalized_pred else 0.0


@registry.asset(
    "metrics",
    "contains",
    desc="预测结果包含参考答案",
    tags=("text",),
    default_aggregation="mean",
)
class ContainsMatchMetric(SimpleMetric):
    """prediction 是否包含 reference。"""

    def compute_value(self, context: MetricContext) -> float:
        label_field = self.args.get("label_field", "label")
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        case_sensitive = bool(self.args.get("case_sensitive", False))
        strip = bool(self.args.get("strip_whitespace", True))

        reference = _extract_field(context, label_field) or ""
        prediction = _extract_field(context, prediction_field) or ""
        reference = _normalize_text(reference, case_sensitive=case_sensitive, strip=strip) or ""
        prediction = _normalize_text(prediction, case_sensitive=case_sensitive, strip=strip) or ""
        return 1.0 if reference and reference in prediction else 0.0


@registry.asset(
    "metrics",
    "numeric_match",
    desc="带容差的数值匹配",
    tags=("numeric",),
    default_aggregation="mean",
)
class NumericMatchMetric(SimpleMetric):
    """针对数值答案的匹配，可配置容差。"""

    def compute_value(self, context: MetricContext) -> float:
        label_field = self.args.get("label_field", "label")
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        tolerance = float(self.args.get("tolerance", 0.0))

        try:
            reference = float(_extract_field(context, label_field))
            prediction = float(_extract_field(context, prediction_field))
        except (TypeError, ValueError):
            return 0.0
        return 1.0 if abs(reference - prediction) <= tolerance else 0.0


@registry.asset(
    "metrics",
    "regex_match",
    desc="基于正则表达式的匹配",
    tags=("text", "regex"),
    default_aggregation="mean",
)
class RegexMatchMetric(SimpleMetric):
    """使用正则进行匹配，可选择匹配 reference 或 prediction。"""

    def setup(self) -> None:
        super().setup()
        pattern = self.args.get("pattern")
        if not pattern:
            raise ValueError("RegexMatchMetric requires 'pattern' in args")
        flags = 0
        if self.args.get("ignore_case", True):
            flags |= re.IGNORECASE
        self._compiled = re.compile(pattern, flags=flags)

    def compute_value(self, context: MetricContext) -> float:
        target_field = self.args.get("target_field", "model_output.answer")
        target = _extract_field(context, target_field) or ""
        return 1.0 if self._compiled.search(str(target)) else 0.0


@registry.asset(
    "metrics",
    "judge_threshold",
    desc="根据裁判分数阈值计算通过率",
    tags=("judge",),
    default_aggregation="mean",
)
class JudgeThresholdMetric(SimpleMetric):
    """将裁判模型分数转换为 0/1。"""

    def compute_value(self, context: MetricContext) -> float:
        judge_field = self.args.get("judge_field", "judge_output.score")
        threshold = float(self.args.get("threshold", 0.5))
        fallback = float(self.args.get("fallback", 0.0))
        value = _extract_field(context, judge_field)
        if value is None:
            return fallback
        try:
            return float(value >= threshold)
        except TypeError:
            return fallback


@registry.asset(
    "metrics",
    "text_length",
    desc="统计文本长度（字符或词）",
    tags=("text", "analysis"),
    default_aggregation="mean",
)
class TextLengthMetric(SimpleMetric):
    """统计文本长度，支持字符或词粒度。"""

    value_key = "length"

    def compute_value(self, context: MetricContext) -> float:
        target_field = self.args.get("target_field", "model_output.answer")
        unit = self.args.get("unit", "char")
        text = _extract_field(context, target_field) or ""
        text = str(text)
        if unit == "word":
            return float(len(text.split()))
        return float(len(text))


@registry.asset(
    "metrics",
    "latency",
    desc="读取模型输出中的延迟信息",
    tags=("latency",),
    default_aggregation="mean",
)
class LatencyMetric(SimpleMetric):
    """读取模型或裁判输出中的延迟信息。"""

    value_key = "latency_ms"

    def compute_value(self, context: MetricContext) -> float:
        field = self.args.get("target_field", "model_output.latency_ms")
        default = float(self.args.get("default_latency", 0.0))
        value = _extract_field(context, field, default=default)
        try:
            return float(value)
        except (TypeError, ValueError):
            return default
