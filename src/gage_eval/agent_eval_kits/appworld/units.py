from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools
from gage_eval.evaluation.support_artifacts import resolve_support_tools


def build_appworld_prompt_context(sample: dict[str, object], runtime_handle: dict[str, object]) -> dict[str, object]:
    """Build AppWorld prompt/runtime context from sample metadata and endpoints."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
    return {
        "instruction": extract_instruction(sample),
        "allowed_apps": list(appworld.get("allowed_apps") or []),
        "task_id": appworld.get("task_id"),
        "ground_truth_mode": appworld.get("ground_truth_mode"),
        "env_endpoint": runtime_handle.get("env_endpoint"),
        "apis_endpoint": runtime_handle.get("apis_endpoint"),
        "mcp_endpoint": runtime_handle.get("mcp_endpoint"),
    }


def build_appworld_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build framework-loop messages for AppWorld."""

    return normalize_messages(sample, fallback_text=extract_instruction(sample))


def build_appworld_tools(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build AppWorld tool schemas."""

    tools = resolve_support_tools(sample)
    if tools:
        return tools
    return normalize_tools(sample)
