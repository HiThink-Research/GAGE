"""MMMU-specific accuracy metric aligned with the legacy llm-eval script."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.registry import registry


def _extract_field(context: MetricContext, descriptor: Optional[str], default: Any = None) -> Any:
    if not descriptor:
        return default
    roots = {
        "sample": context.sample,
        "model_output": context.model_output,
        "judge_output": context.judge_output,
    }
    segments = descriptor.split(".")
    head = segments[0]
    if head in roots:
        base = roots[head]
        tail = ".".join(segments[1:]) if len(segments) > 1 else None
    else:
        base = context.sample
        tail = descriptor
    return _walk(base, tail, default)


def _walk(current: Any, descriptor: Optional[str], default: Any) -> Any:
    if descriptor in (None, ""):
        return current if descriptor in (None, "") else default
    value = current
    for segment in descriptor.split('.'):
        if isinstance(value, Mapping):
            value = value.get(segment)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            try:
                value = value[int(segment)]
            except (ValueError, IndexError):
                return default
        else:
            return default
        if value is None:
            return default
    return value


def _coerce_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if item is not None]
    return [str(value)]


def _clean_text(value: str) -> str:
    return value.strip()


def _is_choice_letter(value: str) -> bool:
    text = value.strip()
    return len(text) == 1 and text.isalpha()


def _match_prediction(prediction: str, target: str) -> bool:
    if not prediction or not target:
        return False
    pred = prediction.strip()
    tgt = target.strip()
    if not tgt:
        return False

    # 优先提取模型最后输出的单字母选项，避免“Option C:”等描述被误判为命中
    if _is_choice_letter(tgt):
        extracted = _extract_choice_letter(pred)
        if extracted:
            return extracted.upper() == tgt.upper()
        # 若未提取到明确字母，则不要用子串匹配，直接按严格比较
        return pred.upper() == tgt.upper()

    # 对非字母标签，保留宽松匹配（原有逻辑）
    if pred == tgt:
        return True
    if tgt in pred:
        return True
    if tgt.lower() in pred.lower():
        return True
    return False


def _extract_choice_letter(prediction: str) -> Optional[str]:
    """从模型输出中提取最后一个明显的选项字母。

    支持模式：
    - \\boxed{B}
    - <answer> B
    - 独立的单字符 A-E（取最后一次出现）
    """

    import re

    patterns = [
        r"\\boxed\{\s*([A-Ea-e])\s*\}",
        r"<answer>\s*([A-Ea-e])",
        r"\b([A-Ea-e])\b",
    ]
    candidates: list[str] = []
    for pat in patterns:
        for match in re.findall(pat, prediction):
            if isinstance(match, tuple):
                for token in match:
                    if token:
                        candidates.append(token)
            elif match:
                candidates.append(match)
    return candidates[-1] if candidates else None


@registry.asset(
    "metrics",
    "mmmu_accuracy",
    desc="MMMU 多模态任务的精确匹配指标（兼容 llm-eval 脚本）。",
    tags=("vision", "mmmu"),
    default_aggregation="mean",
)
class MMMUAccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        label_field = self.args.get("label_field", "sample.choices.0.message.content.0.text")
        prediction_field = self.args.get("prediction_field", "model_output.answer")

        raw_targets = _extract_field(context, label_field, default="")
        targets = [_clean_text(text) for text in _coerce_list(raw_targets) if _clean_text(text)]
        prediction_raw = _extract_field(context, prediction_field, default="")
        prediction = _clean_text(str(prediction_raw))

        matched = any(_match_prediction(prediction, target) for target in targets)
        score = 1.0 if matched else 0.0
        metadata = {
            "prediction": prediction,
            "expected": targets,
        }
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["MMMUAccuracyMetric"]
