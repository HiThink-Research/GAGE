from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_tools
from gage_eval.evaluation.support_artifacts import resolve_support_tools

_APPWORLD_TOOL_USE_CONTRACT = (
    "You are operating inside AppWorld. Use the provided AppWorld tools and endpoints to "
    "inspect the allowed apps and determine the answer. Do not answer from prior knowledge "
    "or refuse access when tools are available."
)


def build_appworld_prompt_context(
    sample: dict[str, object],
    runtime_handle: dict[str, object],
    initialize_result: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build AppWorld prompt/runtime context from sample metadata and endpoints."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    appworld = metadata.get("appworld") if isinstance(metadata.get("appworld"), dict) else {}
    initialize_payload = initialize_result if isinstance(initialize_result, dict) else {}
    instruction = initialize_payload.get("instruction")
    if not isinstance(instruction, str) or not instruction.strip():
        instruction = extract_instruction(sample)
    return {
        "instruction": instruction,
        "allowed_apps": list(appworld.get("allowed_apps") or []),
        "task_id": appworld.get("task_id"),
        "ground_truth_mode": appworld.get("ground_truth_mode"),
        "env_endpoint": runtime_handle.get("env_endpoint"),
        "apis_endpoint": runtime_handle.get("apis_endpoint"),
        "mcp_endpoint": runtime_handle.get("mcp_endpoint"),
    }


def build_appworld_messages(
    sample: dict[str, object],
    *,
    instruction_override: str | None = None,
) -> list[dict[str, object]]:
    """Build framework-loop messages for AppWorld."""

    instruction = ""
    if isinstance(instruction_override, str) and instruction_override.strip():
        instruction = instruction_override.strip()
    else:
        instruction = extract_instruction(sample)
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": _APPWORLD_TOOL_USE_CONTRACT}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": instruction}],
        },
    ]


def build_appworld_tools(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build AppWorld tool schemas."""

    tools = resolve_support_tools(sample)
    if tools:
        return tools
    return normalize_tools(sample)


def build_appworld_instruction(
    sample: dict[str, object],
    *,
    instruction_override: str | None = None,
) -> str:
    """Build the AppWorld instruction with the tool-use contract."""

    if isinstance(instruction_override, str) and instruction_override.strip():
        instruction = instruction_override.strip()
    else:
        instruction = extract_instruction(sample)
    if not instruction:
        return _APPWORLD_TOOL_USE_CONTRACT
    if _APPWORLD_TOOL_USE_CONTRACT in instruction:
        return instruction
    return f"{instruction.rstrip()}\n\nTool-use contract: {_APPWORLD_TOOL_USE_CONTRACT}"
