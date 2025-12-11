"""MathVista accuracy metric for mixed multiple-choice and open-ended QA."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

from gage_eval.metrics.base import MetricContext, MetricResult, SimpleMetric
from gage_eval.metrics.utils import ensure_list_of_strings, extract_field, normalize_text_advanced
from gage_eval.registry import registry


def _extract_choice_letter(prediction: str) -> Optional[str]:
    """从模型输出中提取最后一个明显的选项字母."""

    patterns = [
        r"\\boxed\{\s*([A-Ea-e])\s*\}",
        r"<answer>\s*([A-Ea-e])",
        r"\b([A-Ea-e])\b",
    ]
    candidates: list[str] = []
    for pat in patterns:
        candidates.extend(re.findall(pat, prediction))
    # 展平 tuple
    flat: list[str] = []
    for cand in candidates:
        if isinstance(cand, tuple):
            flat.extend([c for c in cand if c])
        elif cand:
            flat.append(cand)
    return flat[-1] if flat else None


def _resolve_expected_label(
    answer: Any,
    option_map: Dict[str, Any],
    *,
    answer_index_base: int = 0,
) -> Optional[str]:
    if answer is None:
        return None
    # 直接字母
    if isinstance(answer, str) and len(answer.strip()) == 1 and answer.strip().isalpha():
        return answer.strip().upper()
    # 索引
    if isinstance(answer, int):
        labels = list(option_map.keys())
        idx = answer - answer_index_base
        if 0 <= idx < len(labels):
            return str(labels[idx]).upper()
    # 文本匹配
    answer_norm = normalize_text_advanced(str(answer), strip=True, collapse_whitespace=True)
    for label, text in option_map.items():
        if normalize_text_advanced(str(text), strip=True, collapse_whitespace=True) == answer_norm:
            return str(label).upper()
    return None


@registry.asset(
    "metrics",
    "mathvista_accuracy",
    desc="MathVista 混合题型准确率（多选题优先按选项字母匹配，其余按规范化文本匹配）。",
    tags=("vision", "mathvista"),
    default_aggregation="mean",
)
class MathVistaAccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # 配置字段
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        option_map_field = self.args.get("option_map_field", "sample.metadata.option_map")
        correct_choice_field = self.args.get("correct_choice_field", "sample.metadata.correct_choice")
        choices_field = self.args.get("choices_field", "sample.choices")
        answer_field = self.args.get("answer_field", "sample.answer")
        label_field = self.args.get("label_field", "sample.answer")

        # 读取数据
        prediction_raw = extract_field(context, prediction_field, default="")
        prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""
        option_map = extract_field(context, option_map_field, default={}) or {}
        choices = extract_field(context, choices_field, default=[]) or []
        answer = extract_field(context, correct_choice_field)
        if answer is None:
            answer = extract_field(context, answer_field)

        is_multi_choice = bool(option_map) or bool(choices)

        if is_multi_choice:
            # 建立 label -> text 映射
            if not option_map and choices:
                option_map = {c.get("label"): extract_field(c, "message.content.0.text") for c in choices if c}
            option_map = {str(k).upper(): v for k, v in option_map.items() if k is not None}

            expected_label = _resolve_expected_label(answer, option_map, answer_index_base=0)
            pred_label = None
            if prediction:
                extracted = _extract_choice_letter(prediction)
                if extracted:
                    pred_label = extracted.upper()
                else:
                    # 尝试文本匹配
                    pred_norm = normalize_text_advanced(prediction, strip=True, collapse_whitespace=True)
                    for label, text in option_map.items():
                        if normalize_text_advanced(str(text), strip=True, collapse_whitespace=True) == pred_norm:
                            pred_label = label.upper()
                            break
            matched = bool(expected_label and pred_label and expected_label == pred_label)
            score = 1.0 if matched else 0.0
            metadata = {
                "prediction_raw": prediction_raw,
                "prediction_label": pred_label,
                "expected_label": expected_label,
                "option_map": option_map,
            }
            return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)

        # 非多选：按规范化文本精确匹配
        references_raw = extract_field(context, label_field, default="")
        references = [
            normalize_text_advanced(text, strip=True, collapse_whitespace=True)
            for text in ensure_list_of_strings(references_raw)
        ]
        references = [r for r in references if r]
        matched = bool(prediction and references and any(prediction == ref for ref in references))
        score = 1.0 if matched else 0.0
        metadata = {"prediction": prediction, "references": references}
        return MetricResult(sample_id=context.sample_id, values={self.value_key: score}, metadata=metadata)


__all__ = ["MathVistaAccuracyMetric"]
