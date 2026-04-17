from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools

_TERMINAL_COMPLETION_CONTRACT = (
    "Complete the requested workspace changes, then reply with exactly `done` and nothing else."
)


def build_terminal_runtime_context(sample: dict[str, Any]) -> dict[str, Any]:
    """Build the reusable terminal benchmark runtime context."""

    instruction = build_terminal_instruction(sample)
    return {
        "instruction": instruction,
        "cwd": sample.get("cwd") or "/workspace",
        "env": dict(sample.get("env") or {}),
    }


def build_terminal_instruction(sample: dict[str, Any]) -> str:
    """Build the terminal benchmark instruction with completion contract."""

    instruction = extract_instruction(sample)
    if not instruction:
        return _TERMINAL_COMPLETION_CONTRACT
    if _TERMINAL_COMPLETION_CONTRACT in instruction:
        return instruction
    return f"{instruction.rstrip()}\n\nFinal response contract: {_TERMINAL_COMPLETION_CONTRACT}"


def build_terminal_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build framework-loop messages for terminal benchmark samples."""

    messages = normalize_messages(sample, fallback_text=extract_instruction(sample))
    if not messages:
        return [{"role": "system", "content": _TERMINAL_COMPLETION_CONTRACT}]
    return [
        {"role": "system", "content": _TERMINAL_COMPLETION_CONTRACT},
        *messages,
    ]


def build_terminal_tools(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build terminal tool schemas."""

    default_tools = [
        {
            "type": "function",
            "function": {
                "name": "run_shell",
                "description": "Run a shell command in the workspace.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string"},
                        "timeout_s": {"type": "integer"},
                    },
                },
            },
        }
    ]
    return normalize_tools(sample, default_tools)
