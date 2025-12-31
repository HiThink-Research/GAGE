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


def _extract_numeric_answer(prediction: str, answer_type: str) -> Optional[str]:
    """尝试从文本中提取数值答案 (integer/float)"""
    if not prediction:
        return None

    # 1. 尝试直接转换
    try:
        if answer_type == "integer":
            return str(int(float(prediction)))
        elif answer_type == "float":
            return str(float(prediction))
    except (ValueError, TypeError):
        pass

    # 2. 尝试提取最后一个数字
    # 匹配: 可选负号 + 数字 + 可选小数部分
    # 注意: 这种简单的正则可能无法处理千分位逗号等复杂格式，但在 MathVista 语境下通常足够
    numbers = re.findall(r"-?\d+(?:\.\d+)?", prediction)
    if not numbers:
        return None

    last_num = numbers[-1]
    try:
        if answer_type == "integer":
            return str(int(float(last_num)))
        elif answer_type == "float":
            return str(float(last_num))
    except (ValueError, TypeError):
        return None
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
        # 新增 answer_type 字段读取
        answer_type_field = self.args.get("answer_type_field", "sample.answer_type")

        # 读取数据
        prediction_raw = extract_field(context, prediction_field, default="")
        prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""
        option_map = extract_field(context, option_map_field, default={}) or {}
        choices = extract_field(context, choices_field, default=[]) or []
        answer = extract_field(context, correct_choice_field)
        if answer is None:
            answer = extract_field(context, answer_field)
        if answer is None:
            answer = extract_field(context, "sample.label")
        if answer is None:
            answer = extract_field(context, "sample.metadata.answer")
        
        answer_type = extract_field(context, answer_type_field)

        is_multi_choice = bool(option_map) or bool(choices)

        if is_multi_choice:
            # 建立 label -> text 映射
            if not option_map and choices:
                option_map = {c.get("label"): extract_field(c, "message.content.0.text") for c in choices if c}
            option_map = {str(k).upper(): v for k, v in option_map.items() if k is not None}

            expected_label = _resolve_expected_label(answer, option_map, answer_index_base=0)
            pred_label = None
            if prediction and option_map:
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

        # 非多选：如果是数值型，尝试数值提取匹配
        if answer_type in ("integer", "float") and prediction:
            extracted_num = _extract_numeric_answer(prediction, answer_type)
            # 同样尝试提取 reference
            ref_raw = extract_field(context, label_field, default="")
            # 如果 reference 本身就是标准数字，通常不需要正则，但为了保险也走一遍
            ref_str = str(ref_raw)
            ref_num = _extract_numeric_answer(ref_str, answer_type)
            
            if extracted_num is not None and ref_num is not None:
                # 数值匹配 (字符串相等比较即可，因为 _extract_numeric_answer 已经做了规范化)
                matched = (extracted_num == ref_num)
                score = 1.0 if matched else 0.0
                return MetricResult(
                    sample_id=context.sample_id, 
                    values={self.value_key: score}, 
                    metadata={
                        "prediction": prediction, 
                        "extracted_prediction": extracted_num,
                        "reference": ref_num
                    }
                )

        # 非多选且非数值（或提取失败）：回退到按规范化文本精确匹配
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

# 2.benchmark MathVista
@registry.asset(
    "metrics",
    "mathvista_chat_accuracy",
    desc="MathVista 混合题型准确率（多选题优先按选项字母匹配，其余按规范化文本匹配）。",
    tags=("vision", "mathvista"),
    default_aggregation="mean",
)
class MathVistaChataccuracyMetric(SimpleMetric):
    value_key = "acc"

    def compute(self, context: MetricContext) -> MetricResult:
        # 配置字段
        prediction_field = self.args.get("prediction_field", "model_output.answer")
        option_map_field = self.args.get("option_map_field", "sample.metadata.option_map")
        correct_choice_field = self.args.get("correct_choice_field", "sample.metadata.correct_choice")
        choices_field = self.args.get("choices_field", "sample.choices")
        answer_field = self.args.get("answer_field", "sample.answer")
        label_field = self.args.get("label_field", "sample.answer")
        # 新增 answer_type 字段读取
        answer_type_field = self.args.get("answer_type_field", "sample.answer_type")
        question_type_field  = self.args.get("question_type_field", "sample.question_type")
        shot_type_field  = self.args.get("shot_type_field", "sample.shot_type")
        # 读取数据
        answer_type = extract_field(context, answer_type_field)
        shot_type = extract_field(context, shot_type_field)
        question_type = extract_field(context, question_type_field)

        prediction_raw = extract_field(context, prediction_field, default="")

        if shot_type == 'code':
            prediction, error = evaluate_code(prediction_raw)
        else:
            prediction = normalize_text_advanced(str(prediction_raw), strip=True, collapse_whitespace=True) or ""
        option_map = extract_field(context, option_map_field, default={}) or {}
        choices = extract_field(context, choices_field, default=[]) or []
        answer = extract_field(context, correct_choice_field)
        if answer is None:
            answer = extract_field(context, answer_field)
        if answer is None:
            answer = extract_field(context, "sample.label")
        if answer is None:
            answer = extract_field(context, "sample.metadata.answer")


        is_multi_choice = question_type == 'multi_choice'

        if is_multi_choice:
            # 建立 label -> text 映射
            if not option_map and choices:
                option_map = {c.get("label"): extract_field(c, "message.content.0.text") for c in choices if c}
            option_map = {str(k).upper(): v for k, v in option_map.items() if k is not None}

            expected_label = _resolve_expected_label(answer, option_map, answer_index_base=0)
            pred_label = None
            if prediction and option_map:
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

        # 非多选：如果是数值型，尝试数值提取匹配
        if answer_type in ("integer", "float") and prediction:
            extracted_num = _extract_numeric_answer(prediction, answer_type)
            # 同样尝试提取 reference
            ref_raw = extract_field(context, label_field, default="")
            # 如果 reference 本身就是标准数字，通常不需要正则，但为了保险也走一遍
            ref_str = str(ref_raw)
            ref_num = _extract_numeric_answer(ref_str, answer_type)
            
            if extracted_num is not None and ref_num is not None:
                # 数值匹配 (字符串相等比较即可，因为 _extract_numeric_answer 已经做了规范化)
                matched = (extracted_num == ref_num)
                score = 1.0 if matched else 0.0
                return MetricResult(
                    sample_id=context.sample_id, 
                    values={self.value_key: score}, 
                    metadata={
                        "prediction": prediction, 
                        "extracted_prediction": extracted_num,
                        "reference": ref_num
                    }
                )

        # 非多选且非数值（或提取失败）：回退到按规范化文本精确匹配
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


__all__ = ["MathVistaAccuracyMetric", "MathVistaChataccuracyMetric"]
