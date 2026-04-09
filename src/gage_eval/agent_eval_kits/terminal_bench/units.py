from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools


def build_terminal_runtime_context(sample: dict[str, Any]) -> dict[str, Any]:
    """Build the reusable terminal benchmark runtime context."""

    instruction = extract_instruction(sample)
    return {
        "instruction": instruction,
        "cwd": sample.get("cwd") or "/workspace",
        "env": dict(sample.get("env") or {}),
    }


def build_terminal_messages(sample: dict[str, Any]) -> list[dict[str, Any]]:
    """Build framework-loop messages for terminal benchmark samples."""

    return normalize_messages(sample, fallback_text=extract_instruction(sample))


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
