from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools


def build_swebench_runtime_context(sample: dict[str, object]) -> dict[str, object]:
    """Build reusable SWE-bench runtime context."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    return {
        "instruction": extract_instruction(sample),
        "repo": metadata.get("repo"),
        "base_commit": metadata.get("base_commit"),
        "test_command": metadata.get("test_command"),
    }


def build_swebench_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build framework-loop messages for SWE-bench."""

    return normalize_messages(sample, fallback_text=extract_instruction(sample))


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
