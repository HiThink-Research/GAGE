"""Utility functions for SimpleQA Verified preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.utils.mapping import extract_field


def extract_simpleqa_verified_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts and normalizes fields from a raw SimpleQA Verified record."""
    problem = extract_field(record, "problem", default="")
    answer = extract_field(record, "answer", default="")
    topic = extract_field(record, "topic", default="")
    answer_type = extract_field(record, "answer_type", default="")
    multi_step = extract_field(record, "multi_step", default=False)
    requires_reasoning = extract_field(record, "requires_reasoning", default=False)
    urls = extract_field(record, "urls", default="")
    original_index = extract_field(record, "original_index", default=None)
    
    return {
        "problem": problem,
        "answer": answer,
        "topic": topic,
        "answer_type": answer_type,
        "multi_step": multi_step,
        "requires_reasoning": requires_reasoning,
        "urls": urls,
        "original_index": original_index,
    }


def create_prompt(
    record: Dict[str, Any],
    system_prompt: str | None = None,
    instruction: str | None = None,
) -> str:
    """Creates a prompt string for SimpleQA Verified.
    
    Args:
        record: Raw dataset record.
        system_prompt: Optional system prompt to prepend.
        instruction: Optional instruction text to append.
    
    Returns:
        Formatted prompt string.
    """
    problem = extract_field(record, "problem", default="")
    answer_type = extract_field(record, "answer_type", default="")

    prompt_parts: list[str] = []
    # NOTE: System prompt should be passed via the message list (system role),
    # not duplicated inside the user prompt text.

    # Use problem text directly.
    prompt_parts.append(problem.strip())

    # Add per-sample answer-type constraints (SimpleQA Verified-style: direct answer, no guessing).
    normalized_answer_type = str(answer_type or "").strip()
    if normalized_answer_type:
        prompt_parts.append(f"Answer type: {normalized_answer_type}")

    # Instruction is optional and is typically provided via preprocess_kwargs.
    if instruction:
        prompt_parts.append(instruction.strip())

    return "\n\n".join([p for p in prompt_parts if p])
