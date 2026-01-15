"""Utility functions for MATH-500 preprocessor."""

from __future__ import annotations

from typing import Any, Dict

from gage_eval.assets.datasets.utils.mapping import extract_field


def extract_math500_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts and normalizes fields from a raw MATH-500 record."""
    problem = extract_field(record, "problem", default="")
    answer = extract_field(record, "answer", default="")
    solution = extract_field(record, "solution", default="")
    subject = extract_field(record, "subject", default="")
    level = extract_field(record, "level", default=None)
    unique_id = extract_field(record, "unique_id", default=None)
    
    return {
        "problem": problem,
        "answer": answer,
        "solution": solution,
        "subject": subject,
        "level": level,
        "unique_id": unique_id,
    }


def create_prompt(
    record: Dict[str, Any],
    system_prompt: str | None = None,
    instruction: str | None = None,
) -> str:
    """Creates a prompt string for MATH-500.
    
    Args:
        record: Raw dataset record.
        system_prompt: Optional system prompt to prepend.
        instruction: Optional instruction text to append.
    
    Returns:
        Formatted prompt string.
    """
    problem = extract_field(record, "problem", default="")
    
    prompt_parts = []
    if system_prompt:
        prompt_parts.append(system_prompt.strip())
    
    # Use problem text directly without special markers (consistent with MathVista and GPQA)
    prompt_parts.append(problem.strip())
    
    if instruction:
        prompt_parts.append(instruction.strip())
    
    return "\n\n".join(prompt_parts)

