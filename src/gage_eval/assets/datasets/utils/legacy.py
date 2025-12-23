"""Legacy converters and adapters (llm-eval style -> Sample schema)."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.assets.datasets.utils.multimodal import collect_content_fragments, merge_multimodal_inputs
from gage_eval.assets.datasets.utils.mapping import map_question_option_answer

_AUDIT_KEYS = (
    "task_id",
    "version_id",
    "query_id",
    "self_or_open_ai",
    "check_user",
    "check_time",
    "create_at",
    "create_by",
    "review_user",
    "review_time",
)


def convert_llmeval_record(
    record: Dict[str, Any],
    *,
    dataset_id: str,
    dataset_metadata: Optional[Dict[str, Any]] = None,
    question_field: str = "question",
    option_field: str = "choices",
    answer_field: str = "answer",
    answer_index_base: int = 0,
    content_field: str = "messages.0.content",
    content_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a llm-eval style dict into Sample schema (dict)."""

    sample = dict(record)
    sample["_dataset_id"] = dataset_id
    sample["_dataset_metadata"] = dataset_metadata or {}

    # messages/choices/metadata from question-options-answer
    msgs, choices, meta_patch = map_question_option_answer(
        sample,
        question_field=question_field,
        option_field=option_field,
        answer_field=answer_field,
        answer_index_base=answer_index_base,
    )
    if msgs:
        sample["messages"] = msgs
    if choices:
        sample["choices"] = choices
    metadata = dict(sample.get("metadata") or {})
    metadata.update(meta_patch)
    sample["metadata"] = metadata

    # prompt/token_ids → inputs
    inputs: Dict[str, Any] = {}
    if prompt := sample.pop("prompt", None) or sample.get("query"):
        inputs["prompt"] = prompt
    if token_ids := sample.pop("token_ids", None):
        inputs["input_ids"] = token_ids
    if inputs:
        sample["inputs"] = inputs
        sample["chat_template_mode"] = "converted"
        sample["rendered_by"] = "converter"
        sample["template_source"] = "llm-eval"
        sample["cache_suffix"] = "-converted"

    # audit info集中
    audit = {}
    for key in _AUDIT_KEYS:
        if key in sample:
            audit[key] = sample.pop(key)
    if audit:
        sample["audit_info"] = audit

    # 路径/多模态合并
    if content_field:
        fragments = collect_content_fragments(sample, content_field=content_field, content_root=content_root)
        if fragments:
            sample.setdefault("messages", [])
            if not sample["messages"]:
                sample["messages"].append({"role": "user", "content": fragments})
            else:
                sample["messages"][0]["content"] = fragments
        if content_root:
            meta = sample.get("metadata") or {}
            meta["content_root"] = content_root
            meta["content_field"] = content_field
            sample["metadata"] = meta

    merge_multimodal_inputs(sample)
    return sample


__all__ = ["convert_llmeval_record"]
