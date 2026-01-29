"""Utility functions for ARC-AGI-2 preprocessor."""

from __future__ import annotations

from typing import Any, Dict, List

from gage_eval.assets.datasets.utils.mapping import extract_field


def format_grid(grid: List[List[int]]) -> str:
    """Format a 2D grid as a readable string representation.

    Args:
        grid: 2D list of integers representing the grid.

    Returns:
        Formatted string representation of the grid.
    """
    if not grid:
        return "[]"

    lines = []
    for row in grid:
        lines.append(" ".join(str(cell) for cell in row))
    return "\n".join(lines)


def format_grid_as_json(grid: List[List[int]]) -> str:
    """Format a 2D grid as JSON string.

    Args:
        grid: 2D list of integers representing the grid.

    Returns:
        JSON string representation of the grid.
    """
    import json
    return json.dumps(grid)


def extract_arcagi2_fields(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extracts and normalizes fields from a raw ARC-AGI-2 record.

    The ARC-AGI-2 record structure:
    {
        "train": [
            {"input": [[...]], "output": [[...]]},
            ...
        ],
        "test": [
            {"input": [[...]], "output": [[...]]},  # output may be omitted for challenge
        ]
    }

    Args:
        record: Raw dataset record.

    Returns:
        Dict with extracted fields.
    """
    train_examples = extract_field(record, "train", default=[])
    test_data = extract_field(record, "test", default=[])

    # Get first test case (typically there's only one)
    test_case = test_data[0] if test_data else {}
    test_input = extract_field(test_case, "input", default=[])
    test_output = extract_field(test_case, "output", default=None)

    # Extract problem_id from record if available
    problem_id = extract_field(record, "id", default=None)
    if not problem_id:
        # Try alternative field names
        problem_id = extract_field(record, "task_id", default=None)
    if not problem_id:
        # Use hash as fallback
        import json
        problem_id = str(hash(json.dumps(test_input, default=str)))

    return {
        "train_examples": train_examples,
        "test_input": test_input,
        "test_output": test_output,
        "problem_id": problem_id,
    }


def create_prompt(
    record: Dict[str, Any],
    system_prompt: str | None = None,
    instruction: str | None = None,
) -> str:
    """Creates a prompt string for ARC-AGI-2 based on arc-agi-benchmarking format.

    Args:
        record: Raw dataset record.
        system_prompt: Optional system prompt to prepend.
        instruction: Optional instruction text to append.

    Returns:
        Formatted prompt string.
    """
    fields = extract_arcagi2_fields(record)
    train_examples = fields["train_examples"]
    test_input = fields["test_input"]

    prompt_parts = []

    # Add system prompt if provided (but typically handled separately)
    if system_prompt:
        prompt_parts.append(system_prompt.strip())

    # Based on arc-agi-benchmarking system prompt
    prompt_parts.append("You are participating in a puzzle solving competition. You are an expert at solving puzzles.")
    prompt_parts.append("")
    prompt_parts.append("Below is a list of input and output pairs with a pattern. Your goal is to identify the pattern or transformation in the training examples that maps the input to the output, then apply that pattern to the test input to give a final output.")
    prompt_parts.append("")
    prompt_parts.append("Respond in the format of the training output examples")

    # Add training examples
    prompt_parts.append("")
    prompt_parts.append("--Training Examples--")

    if train_examples:
        for i, example in enumerate(train_examples, 1):
            input_grid = format_grid_as_json(example.get("input", []))
            output_grid = format_grid_as_json(example.get("output", []))
            prompt_parts.append(f"Example {i}:")
            prompt_parts.append(f"Input: {input_grid}")
            prompt_parts.append(f"Output: {output_grid}")
            prompt_parts.append("")

    prompt_parts.append("--End of Training Examples--")
    prompt_parts.append("")
    prompt_parts.append("--Test Input--")
    test_input_json = format_grid_as_json(test_input)
    prompt_parts.append(f"Test Input: {test_input_json}")
    prompt_parts.append("--End of Test Input--")
    prompt_parts.append("")
    prompt_parts.append("Your response:")

    if instruction:
        prompt_parts.append(f"\n{instruction.strip()}")

    return "\n".join(prompt_parts)


__all__ = [
    "format_grid",
    "format_grid_as_json",
    "extract_arcagi2_fields",
    "create_prompt",
]