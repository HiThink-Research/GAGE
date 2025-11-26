"""Preprocessor that turns DocVQA-style records into multimodal chat inputs."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Any, Dict, List, Sequence


def convert_sample_to_inputs(
    sample: Dict[str, Any],
    *,
    question_field: str = "question",
    answers_field: str = "choices.0.message.content.0.text",
    image_field: str = "image_path",
    image_root: str | None = None,
    data_path: str | None = None,
    system_prompt: str
    | None = "你是一个能够理解文档图像内容的助手，请基于图像精确回答问题，禁止臆测或虚构信息。",
    instruction: str | None = None,
) -> List[Dict[str, Any]]:
    """Normalize DocVQA samples with text + image content."""

    question = _extract_nested(sample, question_field)
    if question is None:
        raise ValueError(f"DocVQA sample missing question field '{question_field}'")
    answers_raw = _extract_nested(sample, answers_field)
    answers = _parse_answers(answers_raw)
    answers = _enrich_answers_with_choices(question, answers)
    if not answers:
        raise ValueError("DocVQA sample must provide at least one reference answer")

    image_value = _extract_nested(sample, image_field)
    resolved_root = _infer_image_root(image_root, data_path)
    image_url = _resolve_image_url(image_value, resolved_root)

    user_content: List[Dict[str, Any]] = [
        {
            "type": "text",
            "text": _compose_question_prompt(question, instruction),
        }
    ]
    if image_url:
        user_content.append(
            {
                "type": "image_url",
                "image_url": {"url": image_url},
            }
        )

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append(
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": system_prompt.strip(),
                    }
                ],
            }
        )
    messages.append(
        {
            "role": "user",
            "content": user_content,
        }
    )

    sample["messages"] = messages
    sample["prompt"] = question
    metadata = dict(sample.get("metadata") or {})
    metadata.update(
        {
            "answers": answers,
            "question_field": question_field,
            "answers_field": answers_field,
            "image_field": image_field,
        }
    )
    if image_url:
        metadata["image_url"] = image_url
    if resolved_root:
        metadata["image_root"] = resolved_root
    sample["metadata"] = metadata
    return messages


def _extract_nested(sample: Dict[str, Any], field: str | None) -> Any:
    if not field:
        return None

    def _traverse(obj: Any, path: str) -> Any:
        current: Any = obj
        for part in path.split("."):
            if isinstance(current, dict):
                current = current.get(part)
            elif isinstance(current, list):
                try:
                    index = int(part)
                except ValueError:
                    return None
                if index < 0 or index >= len(current):
                    return None
                current = current[index]
            else:
                return None
            if current is None:
                return None
        return current

    for candidate in field.split("|"):
        trimmed = candidate.strip()
        if not trimmed:
            continue
        result = _traverse(sample, trimmed)
        if result is not None:
            return result
    return None


def _parse_answers(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        return [str(item).strip() for item in raw if str(item).strip()]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        try:
            parsed = ast.literal_eval(text)
            if isinstance(parsed, (list, tuple)):
                return [str(item).strip() for item in parsed if str(item).strip()]
        except (ValueError, SyntaxError):
            pass
        return [text]
    return [str(raw).strip()]


def _resolve_image_url(value: Any, image_root: str | None) -> str | None:
    if value in (None, ""):
        return None
    path = str(value)
    if path.startswith("http://") or path.startswith("https://") or path.startswith("file://"):
        return path
    if image_root:
        return str(Path(image_root).joinpath(path).as_posix())
    return path


def _infer_image_root(explicit_root: str | None, data_path: str | None) -> str | None:
    if explicit_root:
        return explicit_root
    if not data_path:
        return None
    try:
        jsonl_path = Path(data_path).expanduser().resolve()
    except (OSError, RuntimeError):
        jsonl_path = Path(data_path).expanduser()
    return str(jsonl_path.parent)


def _compose_question_prompt(question: Any, instruction: str | None) -> str:
    prompt = str(question).strip()
    if instruction:
        return f"{prompt}\n\n{instruction.strip()}"
    return prompt


_CHOICE_PATTERN = re.compile(r"^\s*([A-Z])[\.\)]\s*(.+)$")


def _enrich_answers_with_choices(question: Any, answers: List[str]) -> List[str]:
    if not answers:
        return answers
    choice_map = _extract_choice_map(question)
    if not choice_map:
        return answers
    enriched: List[str] = []
    seen = set()
    for answer in answers:
        normalized = answer.strip()
        if normalized and normalized not in seen:
            enriched.append(normalized)
            seen.add(normalized)
        label = _normalize_choice_label(normalized)
        if label and label in choice_map:
            option_text = choice_map[label]
            if option_text not in seen:
                enriched.append(option_text)
                seen.add(option_text)
    return enriched


def _extract_choice_map(question: Any) -> Dict[str, str]:
    if question is None:
        return {}
    mapping: Dict[str, str] = {}
    for line in str(question).splitlines():
        match = _CHOICE_PATTERN.match(line.strip())
        if match:
            label = match.group(1).upper()
            text = match.group(2).strip()
            if text:
                mapping[label] = text
    return mapping


def _normalize_choice_label(value: str) -> str | None:
    token = value.strip().upper()
    cleaned = token.strip(" .:：()[]{}\"'")
    if len(cleaned) == 1 and cleaned.isalpha():
        return cleaned
    return None


__all__ = ["convert_sample_to_inputs"]
