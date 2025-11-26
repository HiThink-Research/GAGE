"""Multiple-choice accuracy metric aligned with llm-eval logic."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional, Sequence, Tuple

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.registry import registry

_BOXED_PATTERN = re.compile(r"\\boxed\s*\{([^}]*)\}", re.IGNORECASE)
_TOKEN_SPLIT = re.compile(r"[^A-Z]+")


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
    base = roots.get(root_key, context.sample)
    if root_key in roots:
        tail = ".".join(parts[1:]) if len(parts) > 1 else None
    else:
        tail = descriptor
        base = context.sample
    return _walk_mapping(base, tail, default)


def _walk_mapping(source: Any, descriptor: Optional[str], default: Any = None) -> Any:
    if source is None or descriptor in (None, ""):
        return source if descriptor in (None, "") else default
    current = source
    for segment in descriptor.split("."):
        if isinstance(current, Mapping) and segment in current:
            current = current[segment]
            continue
        # 支持以点分隔访问列表索引（如 choices.0.message.content.0.text）
        if isinstance(current, (list, tuple)):
            try:
                idx = int(segment)
            except (TypeError, ValueError):
                return default
            if 0 <= idx < len(current):
                current = current[idx]
                continue
            return default
        return default
    return current


def _normalize_option_map(raw: Any) -> Mapping[str, str]:
    if isinstance(raw, Mapping):
        return {str(key).upper(): str(value) for key, value in raw.items()}
    if isinstance(raw, Sequence):
        normalized = {}
        for entry in raw:
            if isinstance(entry, Mapping):
                label = entry.get("label") or entry.get("index")
                text = entry.get("text") or entry.get("value") or entry.get("content")
                if label is not None and text is not None:
                    normalized[str(label).upper()] = str(text)
        if normalized:
            return normalized
    return {}


def _normalize_choice_label(value: Any, allowed_letters: Sequence[str]) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().upper()
    if not text:
        return None
    direct = text.strip(" .:：()[]{}<>\"'")
    if direct in allowed_letters:
        return direct
    for prefix in ("OPTION", "CHOICE", "ANSWER", "答案", "选项"):
        if direct.startswith(prefix):
            tail = direct[len(prefix) :].strip(" .:：()[]{}<>\"'")
            if tail in allowed_letters:
                return tail
    for token in _TOKEN_SPLIT.split(text):
        if token in allowed_letters:
            return token
    return None


def _match_option_text(value: Any, option_map: Mapping[str, str]) -> Optional[str]:
    if value is None:
        return None
    normalized_value = _normalize_text(value)
    for label, text in option_map.items():
        option_norm = _normalize_text(text)
        if not option_norm:
            continue
        if normalized_value == option_norm or option_norm in normalized_value:
            return label
    return None


def _normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().lower()


def _extract_prediction_letter(raw_answer: Any, allowed_letters: Sequence[str]) -> Optional[str]:
    if raw_answer is None:
        return None
    text = str(raw_answer).strip()
    candidates = []
    matches = _BOXED_PATTERN.findall(text)
    if matches:
        candidates.append(matches[-1])
    candidates.append(text)
    for candidate in candidates:
        normalized = _normalize_choice_label(candidate, allowed_letters)
        if normalized:
            return normalized
    return _normalize_choice_label(text.upper(), allowed_letters)


@registry.asset(
    "metrics",
    "multi_choice_accuracy",
    desc="多选题准确率（对齐 llm-eval 多选逻辑）",
    tags=("text", "multiple-choice"),
    default_aggregation="mean",
)
class MultiChoiceAccuracyMetric(SimpleMetric):
    """Compares extracted option letters with reference answers."""

    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        score, prediction, expected = self._evaluate(context)
        metadata = {}
        if prediction is not None:
            metadata["prediction"] = prediction
        if expected is not None:
            metadata["expected"] = expected
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

    def _evaluate(self, context: MetricContext) -> Tuple[float, Optional[str], Optional[str]]:
        label_field = self.args.get("label_field", "sample.metadata.correct_choice")
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        option_map_field = self.args.get("option_map_field", "sample.metadata.option_map")
        allow_text_match = bool(self.args.get("allow_text_match", True))

        option_map = _normalize_option_map(_extract_field(context, option_map_field, default={}))
        allowed_letters = tuple(option_map.keys()) or tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        expected_raw = _extract_field(context, label_field)
        expected = _normalize_choice_label(expected_raw, allowed_letters) or _match_option_text(
            expected_raw, option_map
        )
        prediction_raw = _extract_field(context, prediction_field, default="")
        prediction = _extract_prediction_letter(prediction_raw, allowed_letters)
        if allow_text_match and not prediction:
            prediction = _match_option_text(prediction_raw, option_map)

        if not expected or not prediction:
            return 0.0, prediction, expected
        return (1.0 if prediction == expected else 0.0), prediction, expected


__all__ = ["MultiChoiceAccuracyMetric"]
