from __future__ import annotations

from gage_eval.agent_eval_kits.common import extract_instruction, normalize_messages, normalize_tools


def build_tau2_prompt_context(sample: dict[str, object], initialize_result: dict[str, object]) -> dict[str, object]:
    """Build the Tau2 runtime-owned prompt context."""

    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    tau2 = metadata.get("tau2") if isinstance(metadata.get("tau2"), dict) else {}
    return {
        "instruction": extract_instruction(sample),
        "domain": tau2.get("domain"),
        "policy": tau2.get("policy"),
        "agent_instruction": tau2.get("agent_instruction"),
        "gage_instruction": tau2.get("gage_instruction"),
        "tools_schema": list(initialize_result.get("tools_schema") or []),
    }


def build_tau2_messages(sample: dict[str, object]) -> list[dict[str, object]]:
    """Build framework-loop Tau2 messages."""

    return normalize_messages(sample, fallback_text=extract_instruction(sample))


def build_tau2_tools(sample: dict[str, object], initialize_result: dict[str, object]) -> list[dict[str, object]]:
    """Build Tau2 tool schemas."""

    return normalize_tools(sample, list(initialize_result.get("tools_schema") or []))
