"""MMAU-Pro 综合评估相关内建 metric。

本文件将官方脚本 `evaluate_mmau_pro_comprehensive.py` 中的三类评估模式，
按照 GAGE 的 metric 接口进行了细粒度封装，方便在 pipeline 中直接使用：

- mmau_pro_closed_nvembed  : 闭卷 / 选择题，基于 NV-Embed-v2 的语义相似度选项匹配
- mmau_pro_aif             : 指令跟随（Audio Instruction Following），基于格式/正则约束检测
- mmau_pro_open_judge      : 开放问答，使用 Qwen2.5 作为 LLM Judge 打分

注意：
- 这些 metric 预期在 MMAU-Pro 的预处理 Sample 上使用，字段来源假定为：
  - 参考答案：sample.label
  - 预测答案：model_output.answer
  - 选项列表：sample.options
  - 任务元信息：sample.metadata.*（由 `MMAUProConverter` 写入）
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np  # 数值运算
import torch  # 模型推理
import torch.nn.functional as F  # 余弦相似度归一化
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer  # 加载 NV-Embed 与 Qwen

from gage_eval.metrics.base import MetricContext, SimpleMetric  # Metric 基类
from gage_eval.metrics.utils import extract_field  # 从上下文读取字段
from gage_eval.registry import registry  # 统一注册入口

import re  # AIF 相关解析
from nltk.tokenize import sent_tokenize  # 句子划分


# ============================================================================
# AIF / 指令跟随评估辅助函数（拷贝自官方脚本，并稍作封装）
# ============================================================================

def _count_words(text: str) -> int:
    """统计单词数量。"""
    return len(str(text).split())


def _count_sentences(text: str) -> int:
    """统计句子数量。"""
    sentences = sent_tokenize(str(text))
    return len(sentences)


def _count_paragraphs(text: str) -> int:
    """统计段落数量，使用 '***' 作为段落分隔符。"""
    paragraphs = str(text).split("***")
    return len([p for p in paragraphs if p.strip()])


def _count_bullet_points(text: str) -> int:
    """统计 markdown 风格的 bullet 数量。"""
    bullets = re.findall(r"(?:^|\n)\s*\*\s+", str(text))
    return len(bullets)


def _count_highlighted_sections(text: str) -> int:
    """统计 *highlight* 形式的高亮片段数量。"""
    highlights = re.findall(r"\*([^*]+)\*", str(text))
    return len(highlights)


def _count_placeholders(text: str) -> int:
    """统计 [placeholder] 样式占位符数量。"""
    placeholders = re.findall(r"\[[^\]]+\]", str(text))
    return len(placeholders)


def _count_capital_words(text: str) -> int:
    """统计全大写单词数量。"""
    words = str(text).split()
    return sum(1 for w in words if w.isupper())


def _count_keyword_frequency(text: str, keyword: str) -> int:
    """统计 keyword 在文本中以完整单词出现的次数。"""
    pattern = r"\b" + re.escape(keyword.lower()) + r"\b"
    matches = re.findall(pattern, str(text).lower())
    return len(matches)


def _has_title(text: str) -> bool:
    """检测是否存在形如 <<title>> 的段落标题。"""
    return bool(re.search(r"<<[^>]+>>", str(text)))


def _has_postscript(text: str, marker: str) -> bool:
    """检测去除非字母字符后，是否包含指定后记 marker。"""
    text_alpha = re.sub(r"[^a-zA-Z]", "", str(text)).lower()
    marker_alpha = re.sub(r"[^a-zA-Z]", "", str(marker)).lower()
    return marker_alpha in text_alpha


def _starts_with_phrase(text: str, phrase: str) -> bool:
    """忽略非字母与大小写，检查是否以指定短语开头。"""
    text_alpha = re.sub(r"[^a-zA-Z ]", "", str(text)).lower()
    phrase_alpha = re.sub(r"[^a-zA-Z ]", "", str(phrase)).lower()
    return text_alpha.startswith(phrase_alpha)


def _ends_with_phrase(text: str, phrase: str) -> bool:
    """忽略非字母与大小写，检查是否以指定短语结尾。"""
    text_alpha = re.sub(r"[^a-zA-Z ]", "", str(text)).lower()
    phrase_alpha = re.sub(r"[^a-zA-Z ]", "", str(phrase)).lower()
    return text_alpha.endswith(phrase_alpha)


def _is_wrapped_in_quotes(text: str) -> bool:
    """检查整体是否被双引号包裹。"""
    stripped = str(text).strip()
    return stripped.startswith('"') and stripped.endswith('"')


def _has_no_commas(text: str) -> bool:
    """检测文本中是否完全没有逗号。"""
    return "," not in str(text)


def _check_sections(text: str, num_sections: int, splitter: str) -> bool:
    """按 splitter 切分后，非空段落数量是否等于 num_sections。"""
    escaped_splitter = re.escape(splitter)
    sections = re.split(rf"\s*{escaped_splitter}\s*", str(text).strip())
    actual_sections = [s for s in sections if s.strip()]
    return len(actual_sections) == num_sections


def _evaluate_aif_sample(response: str, sample_data: Dict[str, Any]) -> bool:
    """针对单条样本执行 AIF 指令跟随校验。"""
    task_identifier = sample_data.get("task_identifier", "") or ""
    kwargs = sample_data.get("kwargs", {}) or {}

    resp = str(response)
    success = False

    if task_identifier == "Include Keywords":
        keywords = kwargs.get("keywords", "").split(", ")
        success = all(k.lower() in resp.lower() for k in keywords if k)

    elif task_identifier == "Keyword Frequency":
        keyword = kwargs.get("keyword", "")
        target = kwargs.get("N", 0)
        actual = _count_keyword_frequency(resp, keyword)
        success = actual == target

    elif task_identifier == "Forbidden Words":
        forbidden_words = kwargs.get("forbidden_words", "").split(", ")
        success = not any(w.lower() in resp.lower() for w in forbidden_words if w)

    elif task_identifier == "Number Paragraphs":
        target = kwargs.get("N", 0)
        actual = _count_paragraphs(resp)
        success = actual == target

    elif task_identifier == "Number Words (at least)":
        target = kwargs.get("N", 0)
        actual = _count_words(resp)
        success = actual >= target

    elif task_identifier == "Number Words (at most)":
        target = kwargs.get("N", 0)
        actual = _count_words(resp)
        success = actual <= target

    elif task_identifier == "Number Words (range)":
        n1 = kwargs.get("N1", 0)
        n2 = kwargs.get("N2", 999)
        actual = _count_words(resp)
        success = n1 <= actual <= n2

    elif task_identifier == "Number Sentences (at least)":
        target = kwargs.get("N", 0)
        actual = _count_sentences(resp)
        success = actual >= target

    elif task_identifier == "Number Sentences (at most)":
        target = kwargs.get("N", 0)
        actual = _count_sentences(resp)
        success = actual <= target

    elif task_identifier == "Number Sentences (range)":
        n1 = kwargs.get("N1", 0)
        n2 = kwargs.get("N2", 999)
        actual = _count_sentences(resp)
        success = n1 <= actual <= n2

    elif task_identifier == "Postscript":
        marker = kwargs.get("postscript_marker", "")
        success = _has_postscript(resp, marker)

    elif task_identifier == "Number Placeholder":
        target = kwargs.get("N", 0)
        actual = _count_placeholders(resp)
        success = actual >= target

    elif task_identifier == "Number Bullets":
        target = kwargs.get("N", 0)
        actual = _count_bullet_points(resp)
        success = actual == target

    elif task_identifier == "Title":
        success = _has_title(resp)

    elif task_identifier == "Minimum Number Highlighted Section":
        target = kwargs.get("N", 0)
        actual = _count_highlighted_sections(resp)
        success = actual >= target

    elif task_identifier == "Multiple Sections":
        target = kwargs.get("N", 0)
        splitter = kwargs.get("section_splitter", "")
        success = _check_sections(resp, target, splitter)

    elif task_identifier == "Repeat Prompt":
        original_prompt = sample_data.get("prompt_transcription", "") or ""
        success = resp.strip().lower().startswith(original_prompt.strip().lower())

    elif task_identifier == "Two Responses":
        separator = "******"
        parts = resp.split(separator)
        success = len(parts) == 2 and parts[0].lower().strip() != parts[1].lower().strip()

    elif task_identifier == "All Uppercase":
        success = resp.isupper()

    elif task_identifier == "All Lowercase":
        success = resp.islower()

    elif task_identifier == "All-capital Words (at least)":
        target = kwargs.get("N", 0)
        actual = _count_capital_words(resp)
        success = actual >= target

    elif task_identifier == "All-capital Words (at most)":
        target = kwargs.get("N", 0)
        actual = _count_capital_words(resp)
        success = actual <= target

    elif task_identifier == "All-capital Words (range)":
        n1 = kwargs.get("N1", 0)
        n2 = kwargs.get("N2", 999)
        actual = _count_capital_words(resp)
        success = n1 <= actual <= n2

    elif task_identifier == "Start Checker":
        phrase = kwargs.get("start_phrase", "")
        success = _starts_with_phrase(resp, phrase)

    elif task_identifier == "End Checker":
        phrase = kwargs.get("end_phrase", "")
        success = _ends_with_phrase(resp, phrase)

    elif task_identifier == "Quotation":
        success = _is_wrapped_in_quotes(resp)

    elif task_identifier == "No Commas":
        success = _has_no_commas(resp)

    return bool(success)


# ============================================================================
# Open-ended / Qwen Judge 相关辅助
# ============================================================================

def _load_qwen_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct") -> Tuple[Any, Any]:
    """加载 Qwen 2.5 评审模型。"""
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    return model, tokenizer


def _create_evaluation_prompt(
    question: str, reference_answer: str, model_response: str, task_type: str
) -> str:
    """构造评审提示词（与官方脚本等价）。"""
    task_context = {
        "sound": "audio content analysis and sound identification",
        "speech": "speech recognition and conversation understanding",
        "music": "music analysis and musical element identification",
        "open": "general open-ended question answering",
    }

    context = task_context.get(task_type, "general question answering")

    prompt = f"""You are an expert evaluator for {context} tasks. Please evaluate the quality of a model's response to a question.

Question: {question}

Reference Answer: {reference_answer}

Model Response: {model_response}

Please evaluate the model response on the following criteria and provide scores from 1-5 (where 5 is best):

1. **Correctness**: How factually accurate is the response compared to the reference?
2. **Relevance**: How well does the response address the specific question asked?
3. **Completeness**: Does the response cover all important aspects mentioned in the reference?
4. **Clarity**: How clear and well-structured is the response?

For each criterion, provide:
- A score from 1-5
- A brief justification (1-2 sentences)

Format your response as:

CORRECTNESS: [score] - [justification]
RELEVANCE: [score] - [justification] 
COMPLETENESS: [score] - [justification]
CLARITY: [score] - [justification]
OVERALL: [average score] - [overall assessment]"""

    return prompt


def _extract_scores_from_evaluation(evaluation_text: str) -> Dict[str, float]:
    """从评审输出文本中解析各项评分。"""
    scores: Dict[str, float] = {}
    patterns = {
        "correctness": r"CORRECTNESS:\s*(\d+)",
        "relevance": r"RELEVANCE:\s*(\d+)",
        "completeness": r"COMPLETENESS:\s*(\d+)",
        "clarity": r"CLARITY:\s*(\d+)",
        "overall": r"OVERALL:\s*(\d+(?:\.\d+)?)",
    }

    for criterion, pattern in patterns.items():
        match = re.search(pattern, evaluation_text, re.IGNORECASE)
        if match:
            scores[criterion] = float(match.group(1))
        else:
            # 找不到时给中性分 3 分
            scores[criterion] = 3.0

    if "overall" not in scores or scores["overall"] == 3.0:
        criteria_scores = [
            scores.get(k, 3.0) for k in ["correctness", "relevance", "completeness", "clarity"]
        ]
        scores["overall"] = float(np.mean(criteria_scores))

    return scores


def _judge_one_with_qwen(
    model: Any,
    tokenizer: Any,
    question: str,
    reference_answer: str,
    model_response: str,
    task_type: str,
) -> Dict[str, float]:
    """对单条开放问答样本进行 LLM Judge 评分。"""
    eval_prompt = _create_evaluation_prompt(question, reference_answer, model_response, task_type)

    messages = [
        {"role": "system", "content": "You are a helpful and objective evaluator."},
        {"role": "user", "content": eval_prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            inputs.input_ids,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.1,
            pad_token_id=tokenizer.eos_token_id,
        )

    # 去掉提示部分，只保留新生成的内容
    new_tokens = [
        output_ids[len(input_ids) :] for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    evaluation_text = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]
    scores = _extract_scores_from_evaluation(evaluation_text)
    return scores


# ============================================================================
# Closed-ended / NV-Embed 相关辅助
# ============================================================================

def _load_nvembed_model(model_name: str = "nvidia/NV-Embed-v2") -> Any:
    """加载 NV-Embed-v2 模型。"""
    model = AutoModel.from_pretrained(model_name, trust_remote_code=True, local_files_only=False)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model


def _match_choice_with_nvembed(
    model: Any, prediction: str, choices: List[str]
) -> Tuple[str, float]:
    """用 NV-Embed-v2 找到与预测语义最接近的选项。"""
    if not choices:
        return "", 0.0

    prediction_embedding = model.encode([prediction], instruction="", max_length=4096)
    prediction_embedding = F.normalize(prediction_embedding, p=2, dim=1)

    choice_embeddings = model.encode(choices, instruction="", max_length=4096)
    choice_embeddings = F.normalize(choice_embeddings, p=2, dim=1)

    scores = (prediction_embedding @ choice_embeddings.T) * 100
    scores = scores.squeeze()

    best_idx = int(torch.argmax(scores).item())
    matched_choice = choices[best_idx]
    confidence = float(torch.max(scores).item())
    return matched_choice, confidence


# ============================================================================
# 内建 metric 定义
# ============================================================================


@registry.asset(
    "metrics",
    "mmau_pro_aif",
    desc="MMAU-Pro 指令跟随 (AIF) 约束检查，通过率型 metric（0/1）",
    tags=("audio", "mmau-pro", "aif"),
    default_aggregation="mean",
)
class MMAUProAIFMetric(SimpleMetric):
    """单样本级 AIF 成功 / 失败检测。"""

    value_key = "success"

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        # 预测文本
        response = extract_field(context, "model_output.answer") or ""

        # 来自预处理 metadata 的任务信息（由 MMAUProConverter 写入）
        task_identifier = extract_field(context, "sample.metadata.task_identifier")
        kwargs = extract_field(context, "sample.metadata.kwargs") or {}
        prompt_transcription = (
            extract_field(context, "sample.metadata.question")
            or extract_field(context, "sample.metadata.transcription")
            or ""
        )

        sample_data = {
            "task_identifier": task_identifier,
            "kwargs": kwargs,
            "prompt_transcription": prompt_transcription,
        }

        success = _evaluate_aif_sample(str(response), sample_data)
        return float(success), {
            "task_identifier": task_identifier,
            "kwargs": kwargs,
        }


@registry.asset(
    "metrics",
    "mmau_pro_open_judge",
    desc="MMAU-Pro 开放问答，Qwen2.5 作为 LLM Judge，返回 overall/5.0 分数",
    tags=("audio", "mmau-pro", "open-ended", "judge"),
    default_aggregation="mean",
)
class MMAUProOpenJudgeMetric(SimpleMetric):
    """对开放式 MMAU-Pro 样本进行 LLM Judge 评分。"""

    value_key = "overall"

    def setup(self) -> None:
        """加载 Qwen 模型（仅在第一次使用时加载一次）。"""
        super().setup()
        model_name = self.args.get("model_name", "Qwen/Qwen2.5-7B-Instruct")
        self._qwen_model, self._qwen_tokenizer = _load_qwen_model(model_name=model_name)

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        question = extract_field(context, "sample.metadata.question") or ""
        reference_answer = extract_field(context, "sample.label") or ""
        model_response = extract_field(context, "model_output.answer") or ""
        task_type = extract_field(context, "sample.metadata.category") or "open"

        if not question or not reference_answer or not model_response:
            # 信息不足时给 0 分并标注跳过
            return 0.0, {
                "skipped": True,
                "reason": "missing_question_or_answer",
            }

        scores = _judge_one_with_qwen(
            self._qwen_model,
            self._qwen_tokenizer,
            str(question),
            str(reference_answer),
            str(model_response),
            str(task_type),
        )

        # official overall 是 1-5，这里归一化到 0-1，方便与其他 metric 聚合
        overall_norm = float(scores.get("overall", 3.0)) / 5.0
        return overall_norm, scores


@registry.asset(
    "metrics",
    "mmau_pro_closed_nvembed",
    desc="MMAU-Pro 闭卷题（选择题），基于 NV-Embed-v2 的语义匹配准确率 (0/1)",
    tags=("audio", "mmau-pro", "closed-ended", "nvembed"),
    default_aggregation="mean",
)
class MMAUProClosedNVEmbedMetric(SimpleMetric):
    """对单条闭卷 / 选择题样本，用 NV-Embed 匹配语义最近的选项并与真实答案比对。"""

    value_key = "correct"

    def setup(self) -> None:
        """加载 NV-Embed-v2 模型。"""
        super().setup()
        model_name = self.args.get("model_name", "nvidia/NV-Embed-v2")
        self._nvembed_model = _load_nvembed_model(model_name=model_name)

    def compute_value(self, context: MetricContext) -> Tuple[float, Dict[str, Any]]:
        # 真实答案与预测文本
        reference = extract_field(context, "sample.label") or ""
        prediction_text = extract_field(context, "model_output.answer") or ""
        # 选项列表，来自 Sample.options
        choices = extract_field(context, "sample.options") or []

        if not reference or not prediction_text or not choices:
            return 0.0, {
                "skipped": True,
                "reason": "missing_reference_prediction_or_choices",
            }

        # 全部转为字符串，避免类型问题
        choices_str = [str(c) for c in choices]
        matched_choice, confidence = _match_choice_with_nvembed(
            self._nvembed_model, str(prediction_text), choices_str
        )

        correct = float(matched_choice == str(reference))
        return correct, {
            "matched_choice": matched_choice,
            "reference": reference,
            "confidence": confidence,
        }


