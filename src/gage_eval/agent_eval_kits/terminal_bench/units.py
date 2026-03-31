"""Terminal benchmark helper units."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    TERMINAL_BENCH_REQUIRED_SURFACES,
    TerminalBenchTaskContext,
)


def _sample_lookup(sample: Mapping[str, Any], *keys: str, default: str = "") -> str:
    for key in keys:
        value = sample.get(key)
        if value not in (None, ""):
            return str(value)
    return default


def _instruction_from_messages(sample: Mapping[str, Any]) -> str:
    messages = sample.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, Mapping):
            continue
        if str(message.get("role") or "").lower() != "user":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            parts: list[str] = []
            for item in content:
                if not isinstance(item, Mapping):
                    continue
                if item.get("type") == "text" and str(item.get("text") or "").strip():
                    parts.append(str(item.get("text")).strip())
            if parts:
                return "\n".join(parts)
    return ""


def _workspace_root(sample: Mapping[str, Any], session: Optional[Any]) -> Optional[str]:
    metadata = sample.get("metadata") or {}
    for key in ("workspace_root", "cwd", "working_dir", "workdir"):
        value = sample.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        metadata_value = metadata.get(key)
        if isinstance(metadata_value, str) and metadata_value.strip():
            return metadata_value.strip()
    for key in ("repo_root", "root", "workspace"):
        metadata_value = metadata.get(key)
        if isinstance(metadata_value, str) and metadata_value.strip():
            return metadata_value.strip()
    if session is not None:
        plan_params = getattr(getattr(session, "plan", None), "params", {})
        if isinstance(plan_params, Mapping):
            for key in ("workspace_root", "cwd", "working_dir", "workdir", "repo_root"):
                value = plan_params.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
    return None


def build_task_context(sample: Mapping[str, Any], session: Optional[Any] = None) -> TerminalBenchTaskContext:
    """Build a normalized task context for a sample."""

    metadata = dict(sample.get("metadata") or {})
    workspace_root = _workspace_root(sample, session)
    instruction = _sample_lookup(sample, "instruction", "goal", "prompt", "task", default="")
    if not instruction:
        instruction = _instruction_from_messages(sample)
    return TerminalBenchTaskContext(
        sample_id=_sample_lookup(sample, "instance_id", "sample_id", "id", default="terminal_bench_sample"),
        instruction=instruction,
        workspace_root=str(workspace_root) if workspace_root is not None else None,
        required_surfaces=TERMINAL_BENCH_REQUIRED_SURFACES,
        metadata=metadata,
    )


def get_instruction(sample: Mapping[str, Any]) -> str:
    """Return the best-effort terminal benchmark instruction."""

    instruction = _sample_lookup(sample, "instruction", "goal", "prompt", "task", default="")
    if instruction:
        return instruction
    return _instruction_from_messages(sample)


def get_sample_id(sample: Mapping[str, Any]) -> str:
    """Return the best-effort sample identifier."""

    return _sample_lookup(sample, "instance_id", "sample_id", "id", default="terminal_bench_sample")
