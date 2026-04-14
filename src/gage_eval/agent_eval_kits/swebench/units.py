from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools

_SWEBENCH_SUBMISSION_CONTRACT = (
    "You must produce the final repository diff as `submission.patch` before finishing. "
    "Use `submit_patch_tool` after applying and testing the fix. Do not finish with prose only."
)


def build_swebench_runtime_context(sample: dict[str, object]) -> dict[str, object]:
    """Build reusable SWE-bench runtime context."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return {
        "instruction": build_swebench_instruction(sample),
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "test_command": metadata.get("test_command"),
    }


def build_swebench_instruction(sample: dict[str, object]) -> str:
    """Build the SWE-bench instruction with submission contract."""

    instruction = extract_instruction(sample)
    if not instruction:
        return _SWEBENCH_SUBMISSION_CONTRACT
    if _SWEBENCH_SUBMISSION_CONTRACT in instruction:
        return instruction
    return f"{instruction.rstrip()}\n\nSubmission contract: {_SWEBENCH_SUBMISSION_CONTRACT}"


def build_swebench_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build framework-loop messages for SWE-bench."""

    messages = normalize_messages(sample, fallback_text=extract_instruction(sample))
    if not messages:
        return [{"role": "system", "content": _SWEBENCH_SUBMISSION_CONTRACT}]
    return [
        {"role": "system", "content": _SWEBENCH_SUBMISSION_CONTRACT},
        *messages,
    ]


def build_swebench_tools(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build the canonical SWE-bench tool surface."""

    default_tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command in the repository workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_s": {"type": "integer"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit_patch_tool",
                "description": "Capture the current git diff as submission.patch.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "timeout_s": {"type": "integer"},
                        "stage_untracked": {"type": "boolean"},
                    },
                },
            },
            "x-gage": {"final_answer_from": "stdout"},
        },
    ]
    return normalize_tools(sample, default_tools)
