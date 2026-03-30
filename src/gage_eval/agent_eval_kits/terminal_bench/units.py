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


def build_task_context(sample: Mapping[str, Any], session: Optional[Any] = None) -> TerminalBenchTaskContext:
    """Build a normalized task context for a sample."""

    metadata = dict(sample.get("metadata") or {})
    workspace_root = sample.get("workspace_root")
    if workspace_root is None and session is not None:
        workspace_root = getattr(getattr(session, "plan", None), "params", {}).get("workspace_root")
    return TerminalBenchTaskContext(
        sample_id=_sample_lookup(sample, "instance_id", "sample_id", "id", default="terminal_bench_sample"),
        instruction=_sample_lookup(sample, "instruction", "goal", "prompt", "task", default=""),
        workspace_root=str(workspace_root) if workspace_root is not None else None,
        required_surfaces=TERMINAL_BENCH_REQUIRED_SURFACES,
        metadata=metadata,
    )


def get_instruction(sample: Mapping[str, Any]) -> str:
    """Return the best-effort terminal benchmark instruction."""

    return _sample_lookup(sample, "instruction", "goal", "prompt", "task", default="")


def get_sample_id(sample: Mapping[str, Any]) -> str:
    """Return the best-effort sample identifier."""

    return _sample_lookup(sample, "instance_id", "sample_id", "id", default="terminal_bench_sample")
